"""Provider-neutral RouteLLM TextMF adapter pinned to the upstream revision.

The upstream training program still requires its PyTorch environment.  This
module freezes the paired-preference input, checkpoint serialization, model
pair adaptation, dev calibration, and exact NumPy inference path used by PAPC.
It deliberately accepts an embedding callback instead of constructing a
provider client.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from papc.baselines.router_accounting import CallReceipt
from papc.baselines.router_contract import (
    CHEAP_MODEL,
    REASONING_EFFORT,
    ROUTELLM,
    ROUTELLM_EMBEDDING_MODEL,
    STRONG_MODEL,
    UPSTREAM_REVISIONS,
    RequestRouterView,
    RouterArtifactBinding,
    SplitManifest,
    assert_disjoint_manifests,
)
from papc.baselines.router_runtime import ScalarPrediction

ROUTELLM_MF_UPSTREAM_REVISION = "0b64fdafe049e596a3f5657c219329f24af24198"
TEXTMF_CHECKPOINT_FORMAT = "papc-routellm-textmf-numpy-v2"
TEXTMF_SERIALIZER_SPEC = (
    "canonical-json:utf8:sort_keys:compact;"
    "fit-embeddings:ordered-example-sha256+little-endian-float32-row-sha256;"
    "float32:P,text_proj.weight,classifier.weight;"
    "no-bias;model-order:gpt-5.4-mini,gpt-5.5"
)
TEXTMF_SERIALIZER_SHA256 = hashlib.sha256(
    TEXTMF_SERIALIZER_SPEC.encode("utf-8")
).hexdigest()
TEXTMF_HIDDEN_SIZE = 128
TEXTMF_TEXT_DIM = 1536
TEXTMF_NUM_CLASSES = 1
TEXTMF_USE_PROJECTION = True
TEXTMF_BATCH_SIZE = 64
TEXTMF_NUM_EPOCHS = 100
TEXTMF_PROMPT_NOISE_ALPHA = 0.1
TEXTMF_LEARNING_RATE = 3e-4
TEXTMF_WEIGHT_DECAY = 1e-5
TEXTMF_RANDOM_SEED = 42
TEXTMF_INNER_TRAIN_FRACTION = 0.95
TEXTMF_OPTIMIZER = "Adam"
TEXTMF_EARLY_STOPPING = False
TEXTMF_CHECKPOINT_SELECTION = "final_epoch"

PAPC_MODEL_IDS: Mapping[str, int] = {
    CHEAP_MODEL: 0,
    STRONG_MODEL: 1,
}

_SHA256_CHARS = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class PairedFitOutcome:
    """One official-fit row before ties are removed."""

    example_sha256: str
    original_request: str = field(repr=False)
    cheap_score: float
    strong_score: float

    def __post_init__(self) -> None:
        _require_sha256(self.example_sha256, "routellm_fit_example")
        if (
            not isinstance(self.original_request, str)
            or not self.original_request.strip()
        ):
            raise ValueError("routellm_fit_request_empty")
        object.__setattr__(
            self,
            "cheap_score",
            _finite_float(self.cheap_score, "routellm_fit_cheap_score"),
        )
        object.__setattr__(
            self,
            "strong_score",
            _finite_float(self.strong_score, "routellm_fit_strong_score"),
        )


@dataclass(frozen=True)
class TextMFPreference:
    """The only label retained for upstream pairwise TextMF training."""

    idx: int
    example_sha256: str
    original_request: str = field(repr=False)
    model_a: str
    model_b: str
    winner: str

    def __post_init__(self) -> None:
        if isinstance(self.idx, bool) or not isinstance(self.idx, int) or self.idx < 0:
            raise ValueError("routellm_preference_idx_invalid")
        _require_sha256(self.example_sha256, "routellm_preference_example")
        if (
            not isinstance(self.original_request, str)
            or not self.original_request.strip()
        ):
            raise ValueError("routellm_preference_request_empty")
        if (self.model_a, self.model_b) != (STRONG_MODEL, CHEAP_MODEL):
            raise ValueError("routellm_preference_model_pair_invalid")
        if self.winner not in {"model_a", "model_b"}:
            raise ValueError("routellm_preference_winner_invalid")

    @property
    def winning_model(self) -> str:
        return self.model_a if self.winner == "model_a" else self.model_b

    def upstream_record(self) -> dict[str, str | int]:
        return {
            "idx": self.idx,
            "model_a": self.model_a,
            "model_b": self.model_b,
            "winner": self.winner,
        }

    def digest_record(self) -> dict[str, str | int]:
        return {
            **self.upstream_record(),
            "example_sha256": self.example_sha256,
            "original_request_sha256": hashlib.sha256(
                self.original_request.encode("utf-8")
            ).hexdigest(),
        }


@dataclass(frozen=True)
class RouteLLMMFFitCorpus:
    fit_registry_sha256: str
    fit_membership_sha256: str
    preferences: tuple[TextMFPreference, ...]
    source_row_count: int
    discarded_tie_count: int

    def __post_init__(self) -> None:
        _require_sha256(self.fit_registry_sha256, "routellm_fit_registry")
        _require_sha256(
            self.fit_membership_sha256,
            "routellm_fit_membership",
        )
        if not self.preferences:
            raise ValueError("routellm_fit_preferences_empty")
        if any(not isinstance(item, TextMFPreference) for item in self.preferences):
            raise TypeError("routellm_fit_preference_invalid")
        if tuple(item.idx for item in self.preferences) != tuple(
            range(len(self.preferences))
        ):
            raise ValueError("routellm_fit_preference_indices_not_contiguous")
        example_ids = tuple(item.example_sha256 for item in self.preferences)
        if len(example_ids) != len(set(example_ids)):
            raise ValueError("routellm_fit_preference_duplicate")
        if (
            isinstance(self.source_row_count, bool)
            or not isinstance(self.source_row_count, int)
            or self.source_row_count <= 0
        ):
            raise ValueError("routellm_fit_source_count_invalid")
        if (
            isinstance(self.discarded_tie_count, bool)
            or not isinstance(self.discarded_tie_count, int)
            or self.discarded_tie_count < 0
        ):
            raise ValueError("routellm_fit_tie_count_invalid")
        if len(self.preferences) + self.discarded_tie_count != self.source_row_count:
            raise ValueError("routellm_fit_count_mismatch")

    @property
    def class_counts(self) -> tuple[tuple[str, int], ...]:
        return (
            (
                "cheap_wins",
                sum(item.winning_model == CHEAP_MODEL for item in self.preferences),
            ),
            (
                "strong_wins",
                sum(item.winning_model == STRONG_MODEL for item in self.preferences),
            ),
        )

    @property
    def class_counts_sha256(self) -> str:
        return _canonical_sha256(dict(self.class_counts))

    @property
    def sha256(self) -> str:
        return _canonical_sha256(
            {
                "fit_registry_sha256": self.fit_registry_sha256,
                "fit_membership_sha256": self.fit_membership_sha256,
                "source_row_count": self.source_row_count,
                "discarded_tie_count": self.discarded_tie_count,
                "preferences": [item.digest_record() for item in self.preferences],
            }
        )

    def upstream_pairwise_records(self) -> tuple[dict[str, str | int], ...]:
        return tuple(item.upstream_record() for item in self.preferences)

    def training_prompts(self) -> tuple[str, ...]:
        """Return prompts in the exact contiguous order expected by ``Q``."""

        return tuple(item.original_request for item in self.preferences)


def build_textmf_fit_corpus(
    outcomes: Sequence[PairedFitOutcome],
    *,
    fit_manifest: SplitManifest,
) -> RouteLLMMFFitCorpus:
    """Drop exact ties and retain only the winner preference used upstream."""

    if isinstance(outcomes, (str, bytes)) or not isinstance(outcomes, Sequence):
        raise TypeError("routellm_fit_outcomes_must_be_sequence")
    if not isinstance(fit_manifest, SplitManifest):
        raise TypeError("routellm_fit_manifest_invalid")
    if fit_manifest.role != "router_fit":
        raise ValueError("routellm_fit_manifest_role_invalid")
    supplied = tuple(outcomes)
    if not supplied or any(not isinstance(item, PairedFitOutcome) for item in supplied):
        raise ValueError("routellm_fit_outcomes_invalid")
    source_ids = tuple(item.example_sha256 for item in supplied)
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("routellm_fit_outcome_duplicate")
    by_id = {item.example_sha256: item for item in supplied}
    _require_exact_membership(
        fit_manifest,
        tuple(by_id),
        field_name="routellm_fit",
    )
    resolved = tuple(by_id[value] for value in fit_manifest.example_sha256s)

    preferences: list[TextMFPreference] = []
    ties = 0
    for outcome in resolved:
        if outcome.strong_score == outcome.cheap_score:
            ties += 1
            continue
        preferences.append(
            TextMFPreference(
                idx=len(preferences),
                example_sha256=outcome.example_sha256,
                original_request=outcome.original_request,
                model_a=STRONG_MODEL,
                model_b=CHEAP_MODEL,
                winner=(
                    "model_a"
                    if outcome.strong_score > outcome.cheap_score
                    else "model_b"
                ),
            )
        )
    return RouteLLMMFFitCorpus(
        fit_registry_sha256=fit_manifest.registry_sha256,
        fit_membership_sha256=_membership_sha256(
            fit_manifest,
            tuple(item.example_sha256 for item in resolved),
        ),
        preferences=tuple(preferences),
        source_row_count=len(resolved),
        discarded_tie_count=ties,
    )


@dataclass(frozen=True)
class RouteLLMMFConfig:
    """The upstream MatrixFactorizationRouter defaults with PAPC model IDs."""

    hidden_size: int = TEXTMF_HIDDEN_SIZE
    text_dim: int = TEXTMF_TEXT_DIM
    num_models: int = 2
    num_classes: int = TEXTMF_NUM_CLASSES
    use_projection: bool = TEXTMF_USE_PROJECTION
    embedding_model: str = ROUTELLM_EMBEDDING_MODEL
    cheap_model: str = CHEAP_MODEL
    cheap_model_id: int = PAPC_MODEL_IDS[CHEAP_MODEL]
    strong_model: str = STRONG_MODEL
    strong_model_id: int = PAPC_MODEL_IDS[STRONG_MODEL]
    batch_size: int = TEXTMF_BATCH_SIZE
    num_epochs: int = TEXTMF_NUM_EPOCHS
    prompt_noise_alpha: float = TEXTMF_PROMPT_NOISE_ALPHA
    learning_rate: float = TEXTMF_LEARNING_RATE
    weight_decay: float = TEXTMF_WEIGHT_DECAY
    random_seed: int = TEXTMF_RANDOM_SEED
    inner_train_fraction: float = TEXTMF_INNER_TRAIN_FRACTION
    optimizer: str = TEXTMF_OPTIMIZER
    early_stopping: bool = TEXTMF_EARLY_STOPPING
    checkpoint_selection: str = TEXTMF_CHECKPOINT_SELECTION
    upstream_revision: str = ROUTELLM_MF_UPSTREAM_REVISION

    def __post_init__(self) -> None:
        expected = {
            "hidden_size": TEXTMF_HIDDEN_SIZE,
            "text_dim": TEXTMF_TEXT_DIM,
            "num_models": 2,
            "num_classes": TEXTMF_NUM_CLASSES,
            "use_projection": TEXTMF_USE_PROJECTION,
            "embedding_model": ROUTELLM_EMBEDDING_MODEL,
            "cheap_model": CHEAP_MODEL,
            "cheap_model_id": 0,
            "strong_model": STRONG_MODEL,
            "strong_model_id": 1,
            "batch_size": TEXTMF_BATCH_SIZE,
            "num_epochs": TEXTMF_NUM_EPOCHS,
            "prompt_noise_alpha": TEXTMF_PROMPT_NOISE_ALPHA,
            "learning_rate": TEXTMF_LEARNING_RATE,
            "weight_decay": TEXTMF_WEIGHT_DECAY,
            "random_seed": TEXTMF_RANDOM_SEED,
            "inner_train_fraction": TEXTMF_INNER_TRAIN_FRACTION,
            "optimizer": TEXTMF_OPTIMIZER,
            "early_stopping": TEXTMF_EARLY_STOPPING,
            "checkpoint_selection": TEXTMF_CHECKPOINT_SELECTION,
            "upstream_revision": ROUTELLM_MF_UPSTREAM_REVISION,
        }
        if self.public_record() != expected:
            raise ValueError("routellm_textmf_config_not_frozen")

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.public_record())

    def public_record(self) -> dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "text_dim": self.text_dim,
            "num_models": self.num_models,
            "num_classes": self.num_classes,
            "use_projection": self.use_projection,
            "embedding_model": self.embedding_model,
            "cheap_model": self.cheap_model,
            "cheap_model_id": self.cheap_model_id,
            "strong_model": self.strong_model,
            "strong_model_id": self.strong_model_id,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "prompt_noise_alpha": self.prompt_noise_alpha,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "random_seed": self.random_seed,
            "inner_train_fraction": self.inner_train_fraction,
            "optimizer": self.optimizer,
            "early_stopping": self.early_stopping,
            "checkpoint_selection": self.checkpoint_selection,
            "upstream_revision": self.upstream_revision,
        }


@dataclass(frozen=True)
class TextMFCheckpoint:
    """Canonical, bias-free export of the upstream TextMF state dict."""

    config: RouteLLMMFConfig
    fit_corpus_sha256: str
    fit_embeddings_sha256: str
    class_counts: tuple[tuple[str, int], ...]
    model_embeddings: np.ndarray = field(repr=False)
    text_projection_weight: np.ndarray = field(repr=False)
    classifier_weight: np.ndarray = field(repr=False)
    serializer_sha256: str = TEXTMF_SERIALIZER_SHA256

    def __post_init__(self) -> None:
        if not isinstance(self.config, RouteLLMMFConfig):
            raise TypeError("routellm_checkpoint_config_invalid")
        _require_sha256(self.fit_corpus_sha256, "routellm_checkpoint_fit_corpus")
        _require_sha256(
            self.fit_embeddings_sha256,
            "routellm_checkpoint_fit_embeddings",
        )
        if self.serializer_sha256 != TEXTMF_SERIALIZER_SHA256:
            raise ValueError("routellm_checkpoint_serializer_mismatch")
        expected_names = ("cheap_wins", "strong_wins")
        if tuple(name for name, _ in self.class_counts) != expected_names:
            raise ValueError("routellm_checkpoint_class_names_invalid")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
            for _, count in self.class_counts
        ):
            raise ValueError("routellm_checkpoint_class_support_invalid")

        model_embeddings = _frozen_float32_array(
            self.model_embeddings,
            shape=(self.config.num_models, self.config.hidden_size),
            field_name="routellm_checkpoint_model_embeddings",
        )
        text_projection = _frozen_float32_array(
            self.text_projection_weight,
            shape=(self.config.hidden_size, self.config.text_dim),
            field_name="routellm_checkpoint_text_projection",
        )
        classifier = _frozen_float32_array(
            self.classifier_weight,
            shape=(self.config.num_classes, self.config.hidden_size),
            field_name="routellm_checkpoint_classifier",
        )
        if np.any(np.linalg.norm(model_embeddings, axis=1) == 0.0):
            raise ValueError("routellm_checkpoint_zero_model_embedding")
        object.__setattr__(self, "model_embeddings", model_embeddings)
        object.__setattr__(self, "text_projection_weight", text_projection)
        object.__setattr__(self, "classifier_weight", classifier)

    @property
    def class_counts_sha256(self) -> str:
        return _canonical_sha256(dict(self.class_counts))

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(
            {
                "format": TEXTMF_CHECKPOINT_FORMAT,
                "serializer_sha256": self.serializer_sha256,
                "config": self.config.public_record(),
                "config_sha256": self.config.sha256,
                "fit_corpus_sha256": self.fit_corpus_sha256,
                "fit_embeddings_sha256": self.fit_embeddings_sha256,
                "class_counts": dict(self.class_counts),
                "class_counts_sha256": self.class_counts_sha256,
                "state_dict": {
                    "P.weight": self.model_embeddings.tolist(),
                    "text_proj.0.weight": self.text_projection_weight.tolist(),
                    "classifier.0.weight": self.classifier_weight.tolist(),
                },
            }
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @classmethod
    def from_bytes(
        cls,
        payload: bytes,
        *,
        expected_sha256: str,
    ) -> TextMFCheckpoint:
        if not isinstance(payload, bytes) or not payload:
            raise TypeError("routellm_checkpoint_payload_invalid")
        _require_sha256(expected_sha256, "routellm_checkpoint_expected")
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ValueError("routellm_checkpoint_sha256_mismatch")
        try:
            record = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("routellm_checkpoint_json_invalid") from exc
        if not isinstance(record, dict):
            raise ValueError("routellm_checkpoint_record_invalid")
        expected_keys = {
            "format",
            "serializer_sha256",
            "config",
            "config_sha256",
            "fit_corpus_sha256",
            "fit_embeddings_sha256",
            "class_counts",
            "class_counts_sha256",
            "state_dict",
        }
        if set(record) != expected_keys:
            raise ValueError("routellm_checkpoint_schema_invalid")
        if record["format"] != TEXTMF_CHECKPOINT_FORMAT:
            raise ValueError("routellm_checkpoint_format_invalid")
        if record["serializer_sha256"] != TEXTMF_SERIALIZER_SHA256:
            raise ValueError("routellm_checkpoint_serializer_mismatch")
        config_record = record["config"]
        if not isinstance(config_record, dict):
            raise ValueError("routellm_checkpoint_config_record_invalid")
        config = RouteLLMMFConfig(**config_record)
        if record["config_sha256"] != config.sha256:
            raise ValueError("routellm_checkpoint_config_sha256_mismatch")
        class_counts_record = record["class_counts"]
        if not isinstance(class_counts_record, dict):
            raise ValueError("routellm_checkpoint_class_counts_invalid")
        class_counts = tuple(
            (name, class_counts_record[name])
            for name in ("cheap_wins", "strong_wins")
            if name in class_counts_record
        )
        if set(class_counts_record) != {"cheap_wins", "strong_wins"}:
            raise ValueError("routellm_checkpoint_class_names_invalid")
        if record["class_counts_sha256"] != _canonical_sha256(class_counts_record):
            raise ValueError("routellm_checkpoint_class_sha256_mismatch")
        state_dict = record["state_dict"]
        if not isinstance(state_dict, dict) or set(state_dict) != {
            "P.weight",
            "text_proj.0.weight",
            "classifier.0.weight",
        }:
            raise ValueError("routellm_checkpoint_state_dict_invalid")
        checkpoint = cls(
            config=config,
            fit_corpus_sha256=record["fit_corpus_sha256"],
            fit_embeddings_sha256=record["fit_embeddings_sha256"],
            class_counts=class_counts,
            model_embeddings=np.asarray(state_dict["P.weight"]),
            text_projection_weight=np.asarray(state_dict["text_proj.0.weight"]),
            classifier_weight=np.asarray(state_dict["classifier.0.weight"]),
        )
        if checkpoint.canonical_bytes != payload:
            raise ValueError("routellm_checkpoint_not_canonical")
        return checkpoint


@dataclass(frozen=True)
class TextMFTrainingState:
    """The three upstream parameters retained by the inference checkpoint."""

    model_embeddings: Any = field(repr=False)
    text_projection_weight: Any = field(repr=False)
    classifier_weight: Any = field(repr=False)
    epochs_completed: int = TEXTMF_NUM_EPOCHS

    def __post_init__(self) -> None:
        if self.epochs_completed != TEXTMF_NUM_EPOCHS:
            raise ValueError("routellm_training_did_not_complete_frozen_epochs")


class TrainTextMF(Protocol):
    def __call__(
        self,
        *,
        corpus: RouteLLMMFFitCorpus,
        prompt_embeddings: np.ndarray,
        config: RouteLLMMFConfig,
        device: str,
    ) -> TextMFTrainingState: ...


def train_and_export_textmf_checkpoint(
    *,
    corpus: RouteLLMMFFitCorpus,
    prompt_embeddings: Mapping[str, Sequence[float]],
    checkpoint_path: Path,
    device: str = "cuda",
    train_backend: TrainTextMF | None = None,
) -> TextMFCheckpoint:
    """Train the pinned fit-only TextMF recipe and write its canonical export.

    ``prompt_embeddings`` must cover exactly the non-tie fit preferences and
    is reordered by their verified corpus order.  External dev and official
    test rows are intentionally absent from this interface.
    """

    if not isinstance(corpus, RouteLLMMFFitCorpus):
        raise TypeError("routellm_training_corpus_invalid")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("routellm_training_device_invalid")
    target = Path(checkpoint_path)
    if not target.name:
        raise ValueError("routellm_checkpoint_path_invalid")
    if os.path.lexists(target):
        raise FileExistsError("routellm_checkpoint_already_exists")
    matrix, embeddings_sha256 = _fit_prompt_embedding_matrix(
        corpus,
        prompt_embeddings,
    )
    config = RouteLLMMFConfig()
    backend = train_backend or _train_textmf_with_torch
    state = backend(
        corpus=corpus,
        prompt_embeddings=matrix,
        config=config,
        device=device,
    )
    if not isinstance(state, TextMFTrainingState):
        raise TypeError("routellm_training_state_invalid")
    checkpoint = TextMFCheckpoint(
        config=config,
        fit_corpus_sha256=corpus.sha256,
        fit_embeddings_sha256=embeddings_sha256,
        class_counts=corpus.class_counts,
        model_embeddings=state.model_embeddings,
        text_projection_weight=state.text_projection_weight,
        classifier_weight=state.classifier_weight,
    )
    _write_new_file_atomic(
        target,
        checkpoint.canonical_bytes,
        field_name="routellm_checkpoint",
    )
    return checkpoint


def _train_textmf_with_torch(
    *,
    corpus: RouteLLMMFFitCorpus,
    prompt_embeddings: np.ndarray,
    config: RouteLLMMFConfig,
    device: str,
) -> TextMFTrainingState:
    """Local reimplementation of the pinned upstream training loop."""

    try:
        import torch
        from torch import nn
        from torch.nn import functional as functional
        from torch.optim import Adam
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("routellm_torch_training_dependency_missing") from exc

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("routellm_cuda_training_device_unavailable")

    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)

    records = list(range(len(corpus.preferences)))
    random.shuffle(records)
    split_at = int(len(records) * config.inner_train_fraction)
    if split_at <= 0 or split_at >= len(records):
        raise ValueError("routellm_inner_fit_split_empty")
    train_indices = records[:split_at]
    validation_indices = records[split_at:]

    winners = []
    losers = []
    for preference in corpus.preferences:
        winner = PAPC_MODEL_IDS[preference.winning_model]
        loser = (
            config.strong_model_id
            if winner == config.cheap_model_id
            else config.cheap_model_id
        )
        winners.append(winner)
        losers.append(loser)
    winner_tensor = torch.tensor(winners, dtype=torch.int64)
    loser_tensor = torch.tensor(losers, dtype=torch.int64)
    prompt_tensor = torch.arange(len(corpus.preferences), dtype=torch.int64)

    def loader(indices: Sequence[int], *, shuffle: bool, batch_size: int) -> Any:
        selected = torch.tensor(indices, dtype=torch.int64)
        dataset = TensorDataset(
            winner_tensor[selected],
            loser_tensor[selected],
            prompt_tensor[selected],
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_loader = loader(
        train_indices,
        shuffle=True,
        batch_size=config.batch_size,
    )
    validation_loader = loader(
        validation_indices,
        shuffle=False,
        batch_size=1024,
    )

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model_embeddings = nn.Embedding(
                config.num_models,
                config.hidden_size,
            )
            self.prompt_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(prompt_embeddings, dtype=torch.float32),
                freeze=True,
            )
            self.text_projection = nn.Linear(
                config.text_dim,
                config.hidden_size,
                bias=False,
            )
            self.classifier = nn.Linear(
                config.hidden_size,
                config.num_classes,
                bias=False,
            )

        def forward(
            self,
            model_win: Any,
            model_loss: Any,
            prompt: Any,
            *,
            test: bool = False,
            alpha: float = 0.05,
        ) -> Any:
            model_win = model_win.to(resolved_device)
            model_loss = model_loss.to(resolved_device)
            prompt = prompt.to(resolved_device)
            winner_embedding = functional.normalize(
                self.model_embeddings(model_win),
                p=2,
                dim=1,
            )
            loser_embedding = functional.normalize(
                self.model_embeddings(model_loss),
                p=2,
                dim=1,
            )
            prompt_embedding = self.prompt_embeddings(prompt)
            if not test:
                prompt_embedding = prompt_embedding + (
                    torch.randn_like(prompt_embedding) * alpha
                )
            prompt_embedding = self.text_projection(prompt_embedding)
            return self.classifier(
                (winner_embedding - loser_embedding) * prompt_embedding
            ).squeeze()

    model = Model().to(resolved_device)
    optimizer = Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_function = nn.BCEWithLogitsLoss(reduction="mean")
    for _ in range(config.num_epochs):
        model.train()
        for model_win, model_loss, prompt in train_loader:
            output = model(
                model_win,
                model_loss,
                prompt,
                alpha=config.prompt_noise_alpha,
            )
            loss = loss_function(output, torch.ones_like(output))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # The pinned upstream evaluator adds its default alpha=0.05 noise.
        # It does not select or stop a checkpoint, but consuming the same RNG
        # stream keeps the subsequent fit epochs deterministic.
        model.eval()
        with torch.no_grad():
            for model_win, model_loss, prompt in validation_loader:
                model(model_win, model_loss, prompt)
                model(model_win, model_loss, prompt, test=True)

    return TextMFTrainingState(
        model_embeddings=(model.model_embeddings.weight.detach().cpu().numpy()),
        text_projection_weight=(model.text_projection.weight.detach().cpu().numpy()),
        classifier_weight=model.classifier.weight.detach().cpu().numpy(),
        epochs_completed=config.num_epochs,
    )


@dataclass(frozen=True)
class RouteLLMMFDevExample:
    example_sha256: str
    predicted_strong_win_rate: float
    cheap_official_quality: float
    strong_official_quality: float
    cheap_cost_usd: float
    strong_cost_usd: float

    def __post_init__(self) -> None:
        _require_sha256(self.example_sha256, "routellm_dev_example")
        for field_name in (
            "predicted_strong_win_rate",
            "cheap_official_quality",
            "strong_official_quality",
        ):
            object.__setattr__(
                self,
                field_name,
                _unit_float(getattr(self, field_name), field_name),
            )
        for field_name in ("cheap_cost_usd", "strong_cost_usd"):
            value = _finite_float(getattr(self, field_name), field_name)
            if value < 0.0:
                raise ValueError(f"{field_name}_negative")
            object.__setattr__(self, field_name, value)

    def public_record(self) -> dict[str, Any]:
        return {
            "example_sha256": self.example_sha256,
            "predicted_strong_win_rate": self.predicted_strong_win_rate,
            "cheap_official_quality": self.cheap_official_quality,
            "strong_official_quality": self.strong_official_quality,
            "cheap_cost_usd": self.cheap_cost_usd,
            "strong_cost_usd": self.strong_cost_usd,
        }


@dataclass(frozen=True)
class RouteLLMMFDevCandidateReceipt:
    threshold: float
    config_sha256: str
    dev_quality: float
    provider_api_cost_usd: float
    strong_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "threshold",
            _unit_float(self.threshold, "routellm_dev_threshold"),
        )
        if self.config_sha256 != _canonical_sha256({"route_threshold": self.threshold}):
            raise ValueError("routellm_dev_config_sha256_mismatch")
        object.__setattr__(
            self,
            "dev_quality",
            _unit_float(self.dev_quality, "routellm_dev_quality"),
        )
        cost = _finite_float(
            self.provider_api_cost_usd,
            "routellm_dev_provider_api_cost",
        )
        if cost < 0.0:
            raise ValueError("routellm_dev_provider_api_cost_negative")
        object.__setattr__(self, "provider_api_cost_usd", cost)
        if (
            isinstance(self.strong_count, bool)
            or not isinstance(self.strong_count, int)
            or self.strong_count < 0
        ):
            raise ValueError("routellm_dev_strong_count_invalid")

    def public_record(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "config_sha256": self.config_sha256,
            "dev_quality": self.dev_quality,
            "provider_api_cost_usd": self.provider_api_cost_usd,
            "strong_count": self.strong_count,
        }


@dataclass(frozen=True)
class RouteLLMMFDevCalibration:
    dev_registry_sha256: str
    dev_membership_sha256: str
    strong_full_dev_anchor: float
    threshold: float
    dev_row_count: int
    observed_strong_count: int
    dev_examples_sha256: str
    candidate_receipts: tuple[RouteLLMMFDevCandidateReceipt, ...]
    selection_rule: str = "strong_full_dev_anchor_then_min_provider_cost_then_min_strong_calls_then_config_sha256"

    def __post_init__(self) -> None:
        _require_sha256(self.dev_registry_sha256, "routellm_dev_registry")
        _require_sha256(
            self.dev_membership_sha256,
            "routellm_dev_membership",
        )
        object.__setattr__(
            self,
            "strong_full_dev_anchor",
            _unit_float(
                self.strong_full_dev_anchor,
                "routellm_strong_full_dev_anchor",
            ),
        )
        object.__setattr__(
            self,
            "threshold",
            _unit_float(self.threshold, "routellm_dev_threshold"),
        )
        if (
            isinstance(self.dev_row_count, bool)
            or not isinstance(self.dev_row_count, int)
            or self.dev_row_count <= 0
        ):
            raise ValueError("routellm_dev_row_count_invalid")
        if (
            isinstance(self.observed_strong_count, bool)
            or not isinstance(self.observed_strong_count, int)
            or not 0 <= self.observed_strong_count <= self.dev_row_count
        ):
            raise ValueError("routellm_dev_strong_count_invalid")
        _require_sha256(self.dev_examples_sha256, "routellm_dev_examples")
        receipts = tuple(self.candidate_receipts)
        if not receipts or any(
            not isinstance(item, RouteLLMMFDevCandidateReceipt) for item in receipts
        ):
            raise ValueError("routellm_dev_candidate_receipts_invalid")
        if any(item.strong_count > self.dev_row_count for item in receipts):
            raise ValueError("routellm_dev_candidate_strong_count_invalid")
        if not any(item.strong_count == self.dev_row_count for item in receipts):
            raise ValueError("routellm_dev_all_strong_candidate_required")
        object.__setattr__(self, "candidate_receipts", receipts)
        if (
            self.selection_rule
            != "strong_full_dev_anchor_then_min_provider_cost_then_min_strong_calls_then_config_sha256"
        ):
            raise ValueError("routellm_dev_selection_rule_invalid")

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.public_record())

    def public_record(self) -> dict[str, Any]:
        return {
            "dev_registry_sha256": self.dev_registry_sha256,
            "dev_membership_sha256": self.dev_membership_sha256,
            "strong_full_dev_anchor": self.strong_full_dev_anchor,
            "threshold": self.threshold,
            "dev_row_count": self.dev_row_count,
            "observed_strong_count": self.observed_strong_count,
            "dev_examples_sha256": self.dev_examples_sha256,
            "candidate_receipts": [
                item.public_record() for item in self.candidate_receipts
            ],
            "selection_rule": self.selection_rule,
        }


def freeze_dev_threshold(
    examples: Sequence[RouteLLMMFDevExample],
    *,
    dev_manifest: SplitManifest,
) -> RouteLLMMFDevCalibration:
    """Select a dev threshold without a caller-chosen routing rate."""

    if isinstance(examples, (str, bytes)) or not isinstance(examples, Sequence):
        raise TypeError("routellm_dev_examples_must_be_sequence")
    if not isinstance(dev_manifest, SplitManifest):
        raise TypeError("routellm_dev_manifest_invalid")
    if dev_manifest.role != "router_dev":
        raise ValueError("routellm_dev_manifest_role_invalid")
    supplied = tuple(examples)
    if not supplied or any(
        not isinstance(item, RouteLLMMFDevExample) for item in supplied
    ):
        raise ValueError("routellm_dev_examples_invalid")
    source_ids = tuple(item.example_sha256 for item in supplied)
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("routellm_dev_example_duplicate")
    by_id = {item.example_sha256: item for item in supplied}
    _require_exact_membership(
        dev_manifest,
        tuple(by_id),
        field_name="routellm_dev",
    )
    resolved = tuple(by_id[value] for value in dev_manifest.example_sha256s)
    quality_floor = sum(item.strong_official_quality for item in resolved) / len(
        resolved
    )
    thresholds = tuple(
        sorted(
            {
                0.0,
                1.0,
                *(item.predicted_strong_win_rate for item in resolved),
            }
        )
    )
    receipts: list[RouteLLMMFDevCandidateReceipt] = []
    for threshold in thresholds:
        strong_flags = tuple(
            item.predicted_strong_win_rate >= threshold for item in resolved
        )
        receipts.append(
            RouteLLMMFDevCandidateReceipt(
                threshold=threshold,
                config_sha256=_canonical_sha256({"route_threshold": threshold}),
                dev_quality=sum(
                    item.strong_official_quality
                    if use_strong
                    else item.cheap_official_quality
                    for item, use_strong in zip(resolved, strong_flags)
                )
                / len(resolved),
                provider_api_cost_usd=sum(
                    item.strong_cost_usd if use_strong else item.cheap_cost_usd
                    for item, use_strong in zip(resolved, strong_flags)
                ),
                strong_count=sum(strong_flags),
            )
        )
    feasible = tuple(
        item for item in receipts if item.dev_quality + 1e-12 >= quality_floor
    )
    if not feasible:
        raise ValueError("routellm_no_dev_candidate_meets_quality_floor")
    selected = min(
        feasible,
        key=lambda item: (
            item.provider_api_cost_usd,
            item.strong_count,
            item.config_sha256,
        ),
    )
    return RouteLLMMFDevCalibration(
        dev_registry_sha256=dev_manifest.registry_sha256,
        dev_membership_sha256=_membership_sha256(
            dev_manifest,
            tuple(item.example_sha256 for item in resolved),
        ),
        strong_full_dev_anchor=quality_floor,
        threshold=selected.threshold,
        dev_row_count=len(resolved),
        observed_strong_count=selected.strong_count,
        dev_examples_sha256=_canonical_sha256(
            [item.public_record() for item in resolved]
        ),
        candidate_receipts=tuple(receipts),
    )


@dataclass(frozen=True)
class RouteLLMMFArtifact:
    surface: str
    fit_registry_sha256: str
    dev_registry_sha256: str
    test_registry_sha256: str
    fit_corpus_sha256: str
    fit_membership_sha256: str
    dev_membership_sha256: str
    split_binding_sha256: str
    fit_row_count: int
    discarded_tie_count: int
    class_counts: tuple[tuple[str, int], ...]
    class_counts_sha256: str
    checkpoint_sha256: str
    config_sha256: str
    serializer_sha256: str
    dev_calibration: RouteLLMMFDevCalibration
    upstream_revision: str = ROUTELLM_MF_UPSTREAM_REVISION
    cheap_model: str = CHEAP_MODEL
    strong_model: str = STRONG_MODEL
    reasoning_effort: str = REASONING_EFFORT
    embedding_model: str = ROUTELLM_EMBEDDING_MODEL

    def __post_init__(self) -> None:
        if not isinstance(self.surface, str) or not self.surface.strip():
            raise ValueError("routellm_artifact_surface_invalid")
        for field_name in (
            "fit_registry_sha256",
            "dev_registry_sha256",
            "test_registry_sha256",
            "fit_corpus_sha256",
            "fit_membership_sha256",
            "dev_membership_sha256",
            "split_binding_sha256",
            "class_counts_sha256",
            "checkpoint_sha256",
            "config_sha256",
            "serializer_sha256",
        ):
            _require_sha256(getattr(self, field_name), f"routellm_{field_name}")
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
            raise ValueError("routellm_artifact_split_hash_reused")
        if (
            isinstance(self.fit_row_count, bool)
            or not isinstance(self.fit_row_count, int)
            or self.fit_row_count <= 0
        ):
            raise ValueError("routellm_artifact_fit_count_invalid")
        if (
            isinstance(self.discarded_tie_count, bool)
            or not isinstance(self.discarded_tie_count, int)
            or self.discarded_tie_count < 0
        ):
            raise ValueError("routellm_artifact_tie_count_invalid")
        if tuple(name for name, _ in self.class_counts) != (
            "cheap_wins",
            "strong_wins",
        ):
            raise ValueError("routellm_artifact_class_names_invalid")
        if sum(count for _, count in self.class_counts) != self.fit_row_count:
            raise ValueError("routellm_artifact_class_count_mismatch")
        if self.class_counts_sha256 != _canonical_sha256(dict(self.class_counts)):
            raise ValueError("routellm_artifact_class_sha256_mismatch")
        if self.config_sha256 != RouteLLMMFConfig().sha256:
            raise ValueError("routellm_artifact_config_sha256_mismatch")
        if self.serializer_sha256 != TEXTMF_SERIALIZER_SHA256:
            raise ValueError("routellm_artifact_serializer_mismatch")
        if (
            not isinstance(self.dev_calibration, RouteLLMMFDevCalibration)
            or self.dev_calibration.dev_registry_sha256 != self.dev_registry_sha256
        ):
            raise ValueError("routellm_artifact_dev_calibration_mismatch")
        if self.dev_membership_sha256 != self.dev_calibration.dev_membership_sha256:
            raise ValueError("routellm_artifact_dev_membership_mismatch")
        if self.upstream_revision != ROUTELLM_MF_UPSTREAM_REVISION:
            raise ValueError("routellm_artifact_upstream_revision_mismatch")
        if self.upstream_revision != UPSTREAM_REVISIONS[ROUTELLM]:
            raise ValueError("routellm_artifact_contract_revision_mismatch")
        if (self.cheap_model, self.strong_model) != (
            CHEAP_MODEL,
            STRONG_MODEL,
        ):
            raise ValueError("routellm_artifact_model_pair_mismatch")
        if self.reasoning_effort != REASONING_EFFORT:
            raise ValueError("routellm_artifact_reasoning_effort_mismatch")
        if self.embedding_model != ROUTELLM_EMBEDDING_MODEL:
            raise ValueError("routellm_artifact_embedding_model_mismatch")

    @property
    def artifact_sha256(self) -> str:
        return _canonical_sha256(self.public_record())

    @property
    def threshold(self) -> float:
        return self.dev_calibration.threshold

    def public_record(self) -> dict[str, Any]:
        return {
            "method": ROUTELLM,
            "surface": self.surface,
            "fit_registry_sha256": self.fit_registry_sha256,
            "dev_registry_sha256": self.dev_registry_sha256,
            "test_registry_sha256": self.test_registry_sha256,
            "fit_corpus_sha256": self.fit_corpus_sha256,
            "fit_membership_sha256": self.fit_membership_sha256,
            "dev_membership_sha256": self.dev_membership_sha256,
            "split_binding_sha256": self.split_binding_sha256,
            "fit_row_count": self.fit_row_count,
            "discarded_tie_count": self.discarded_tie_count,
            "class_counts": dict(self.class_counts),
            "class_counts_sha256": self.class_counts_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_sha256": self.config_sha256,
            "serializer_sha256": self.serializer_sha256,
            "dev_calibration": self.dev_calibration.public_record(),
            "dev_calibration_sha256": self.dev_calibration.sha256,
            "upstream_revision": self.upstream_revision,
            "cheap_model": self.cheap_model,
            "strong_model": self.strong_model,
            "reasoning_effort": self.reasoning_effort,
            "embedding_model": self.embedding_model,
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


def bind_routellm_mf_artifact(
    *,
    surface: str,
    corpus: RouteLLMMFFitCorpus,
    checkpoint: TextMFCheckpoint,
    fit_manifest: SplitManifest,
    dev_manifest: SplitManifest,
    test_manifest: SplitManifest,
    dev_calibration: RouteLLMMFDevCalibration,
) -> RouteLLMMFArtifact:
    if not isinstance(corpus, RouteLLMMFFitCorpus):
        raise TypeError("routellm_artifact_fit_corpus_invalid")
    if not isinstance(checkpoint, TextMFCheckpoint):
        raise TypeError("routellm_artifact_checkpoint_invalid")
    if checkpoint.fit_corpus_sha256 != corpus.sha256:
        raise ValueError("routellm_artifact_checkpoint_fit_mismatch")
    if checkpoint.class_counts != corpus.class_counts:
        raise ValueError("routellm_artifact_checkpoint_classes_mismatch")
    assert_disjoint_manifests(fit_manifest, dev_manifest, test_manifest)
    if corpus.fit_registry_sha256 != fit_manifest.registry_sha256:
        raise ValueError("routellm_artifact_fit_registry_mismatch")
    if corpus.fit_membership_sha256 != _membership_sha256(
        fit_manifest,
        fit_manifest.example_sha256s,
    ):
        raise ValueError("routellm_artifact_fit_membership_mismatch")
    if dev_calibration.dev_registry_sha256 != dev_manifest.registry_sha256:
        raise ValueError("routellm_artifact_calibration_registry_mismatch")
    return RouteLLMMFArtifact(
        surface=surface,
        fit_registry_sha256=corpus.fit_registry_sha256,
        dev_registry_sha256=dev_manifest.registry_sha256,
        test_registry_sha256=test_manifest.registry_sha256,
        fit_corpus_sha256=corpus.sha256,
        fit_membership_sha256=corpus.fit_membership_sha256,
        dev_membership_sha256=dev_calibration.dev_membership_sha256,
        split_binding_sha256=_canonical_sha256(
            {
                "fit": fit_manifest.registry_sha256,
                "dev": dev_manifest.registry_sha256,
                "test": test_manifest.registry_sha256,
                "fit_examples": fit_manifest.example_sha256s,
                "dev_examples": dev_manifest.example_sha256s,
                "test_examples": test_manifest.example_sha256s,
            }
        ),
        fit_row_count=len(corpus.preferences),
        discarded_tie_count=corpus.discarded_tie_count,
        class_counts=corpus.class_counts,
        class_counts_sha256=corpus.class_counts_sha256,
        checkpoint_sha256=checkpoint.sha256,
        config_sha256=checkpoint.config.sha256,
        serializer_sha256=checkpoint.serializer_sha256,
        dev_calibration=dev_calibration,
    )


@dataclass(frozen=True)
class RouteLLMMFEmbeddingRequest:
    prompt: str = field(repr=False)
    model: str = ROUTELLM_EMBEDDING_MODEL

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("routellm_embedding_prompt_empty")
        if self.model != ROUTELLM_EMBEDDING_MODEL:
            raise ValueError("routellm_embedding_model_not_frozen")

    @property
    def sha256(self) -> str:
        return _canonical_sha256(
            {
                "model": self.model,
                "prompt_sha256": hashlib.sha256(
                    self.prompt.encode("utf-8")
                ).hexdigest(),
            }
        )


@dataclass(frozen=True)
class RouteLLMMFEmbeddingResult:
    request_sha256: str
    embedding: tuple[float, ...] = field(repr=False)
    receipt: CallReceipt

    def __post_init__(self) -> None:
        _require_sha256(self.request_sha256, "routellm_embedding_request")
        if not isinstance(self.embedding, tuple):
            object.__setattr__(self, "embedding", tuple(self.embedding))
        if len(self.embedding) != TEXTMF_TEXT_DIM:
            raise ValueError("routellm_embedding_dimension_mismatch")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in self.embedding
        ):
            raise ValueError("routellm_embedding_value_invalid")
        if not isinstance(self.receipt, CallReceipt):
            raise TypeError("routellm_embedding_receipt_invalid")
        if (
            self.receipt.role != "router"
            or self.receipt.phase != "route_embedding"
            or self.receipt.model != ROUTELLM_EMBEDDING_MODEL
            or self.receipt.provider_calls <= 0
        ):
            raise ValueError("routellm_embedding_provider_receipt_invalid")


class EmbedPrompt(Protocol):
    def __call__(
        self,
        request: RouteLLMMFEmbeddingRequest,
    ) -> RouteLLMMFEmbeddingResult: ...


class TextMFNumpyModel:
    """Inference-equivalent mirror of upstream ``MFModel.pred_win_rate``."""

    def __init__(self, checkpoint: TextMFCheckpoint) -> None:
        if not isinstance(checkpoint, TextMFCheckpoint):
            raise TypeError("routellm_textmf_checkpoint_invalid")
        self._checkpoint = checkpoint

    def predict_strong_win_rate(self, embedding: Sequence[float]) -> float:
        prompt_embedding = np.asarray(embedding, dtype=np.float32)
        if prompt_embedding.shape != (self._checkpoint.config.text_dim,):
            raise ValueError("routellm_embedding_dimension_mismatch")
        if not bool(np.all(np.isfinite(prompt_embedding))):
            raise ValueError("routellm_embedding_value_invalid")

        ids = (
            self._checkpoint.config.strong_model_id,
            self._checkpoint.config.cheap_model_id,
        )
        model_embeddings = self._checkpoint.model_embeddings[list(ids)]
        model_embeddings = model_embeddings / np.linalg.norm(
            model_embeddings,
            axis=1,
            keepdims=True,
        )
        projected_prompt = self._checkpoint.text_projection_weight @ prompt_embedding
        logits = (
            model_embeddings * projected_prompt
        ) @ self._checkpoint.classifier_weight[0]
        delta = float(logits[0] - logits[1])
        if delta >= 0.0:
            return 1.0 / (1.0 + math.exp(-delta))
        exponential = math.exp(delta)
        return exponential / (1.0 + exponential)


@dataclass
class RouteLLMMFPredictor:
    artifact: RouteLLMMFArtifact
    checkpoint: TextMFCheckpoint
    embed_prompt: EmbedPrompt
    _model: TextMFNumpyModel = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, RouteLLMMFArtifact):
            raise TypeError("routellm_predictor_artifact_invalid")
        if not isinstance(self.checkpoint, TextMFCheckpoint):
            raise TypeError("routellm_predictor_checkpoint_invalid")
        if self.checkpoint.sha256 != self.artifact.checkpoint_sha256:
            raise ValueError("routellm_predictor_checkpoint_sha256_mismatch")
        if self.checkpoint.config.sha256 != self.artifact.config_sha256:
            raise ValueError("routellm_predictor_config_sha256_mismatch")
        if self.checkpoint.class_counts_sha256 != self.artifact.class_counts_sha256:
            raise ValueError("routellm_predictor_class_sha256_mismatch")
        if not callable(self.embed_prompt):
            raise TypeError("routellm_predictor_embedder_not_callable")
        self._model = TextMFNumpyModel(self.checkpoint)

    @property
    def threshold(self) -> float:
        return self.artifact.threshold

    def __call__(self, view: RequestRouterView) -> ScalarPrediction:
        if not isinstance(view, RequestRouterView):
            raise TypeError("routellm_predictor_request_view_required")
        request = RouteLLMMFEmbeddingRequest(prompt=view.original_request)
        embedded = self.embed_prompt(request)
        if not isinstance(embedded, RouteLLMMFEmbeddingResult):
            raise TypeError("routellm_embedding_result_invalid")
        if embedded.request_sha256 != request.sha256:
            raise ValueError("routellm_embedding_request_sha256_mismatch")
        started = time.perf_counter()
        win_rate = self._model.predict_strong_win_rate(embedded.embedding)
        latency = max(0.0, time.perf_counter() - started)
        return ScalarPrediction(
            value=win_rate,
            receipts=(
                embedded.receipt,
                CallReceipt(
                    role="router",
                    phase="route_mf",
                    model=ROUTELLM,
                    reasoning_effort="local",
                    provider_calls=0,
                    input_tokens=0,
                    cached_input_tokens=0,
                    output_tokens=0,
                    reasoning_tokens=0,
                    retry_count=0,
                    latency_seconds=latency,
                    provider_reported_cost_usd=0.0,
                ),
            ),
        )


def load_routellm_mf_predictor(
    *,
    artifact: RouteLLMMFArtifact,
    checkpoint_payload: bytes,
    embed_prompt: Callable[
        [RouteLLMMFEmbeddingRequest],
        RouteLLMMFEmbeddingResult,
    ],
) -> RouteLLMMFPredictor:
    checkpoint = TextMFCheckpoint.from_bytes(
        checkpoint_payload,
        expected_sha256=artifact.checkpoint_sha256,
    )
    return RouteLLMMFPredictor(
        artifact=artifact,
        checkpoint=checkpoint,
        embed_prompt=embed_prompt,
    )


def upstream_training_dependency_blockers(
    *,
    device: str = "cuda",
) -> tuple[str, ...]:
    """Report local dependencies for the fit-only adapted trainer."""

    if not isinstance(device, str) or not device.strip():
        raise ValueError("routellm_training_device_invalid")
    if importlib.util.find_spec("torch") is None:
        return ("missing_training_dependency:torch",)
    if device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            return ("training_device_unavailable:cuda",)
    return ()


def _fit_prompt_embedding_matrix(
    corpus: RouteLLMMFFitCorpus,
    prompt_embeddings: Mapping[str, Sequence[float]],
) -> tuple[np.ndarray, str]:
    if not isinstance(prompt_embeddings, Mapping):
        raise TypeError("routellm_fit_embeddings_must_be_mapping")
    expected = tuple(preference.example_sha256 for preference in corpus.preferences)
    if set(prompt_embeddings) != set(expected):
        raise ValueError("routellm_fit_embedding_membership_mismatch")
    rows = []
    digest_rows = []
    for example_sha256 in expected:
        row = _frozen_float32_array(
            prompt_embeddings[example_sha256],
            shape=(TEXTMF_TEXT_DIM,),
            field_name="routellm_fit_embedding",
        ).astype("<f4", copy=False)
        rows.append(row)
        digest_rows.append(
            {
                "example_sha256": example_sha256,
                "embedding_float32_sha256": hashlib.sha256(
                    row.tobytes(order="C")
                ).hexdigest(),
            }
        )
    matrix = np.stack(rows).astype(np.float32, copy=False)
    matrix.setflags(write=False)
    return matrix, _canonical_sha256(
        {
            "embedding_model": ROUTELLM_EMBEDDING_MODEL,
            "fit_corpus_sha256": corpus.sha256,
            "rows": digest_rows,
        }
    )


def _write_new_file_atomic(
    path: Path,
    payload: bytes,
    *,
    field_name: str,
) -> None:
    target = Path(path)
    if not target.name:
        raise ValueError(f"{field_name}_path_invalid")
    if os.path.lexists(target):
        raise FileExistsError(f"{field_name}_already_exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{hashlib.sha256(payload).hexdigest()}.tmp"
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if os.path.lexists(temporary):
            temporary.unlink()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


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
        field_name="routellm_membership",
    )
    return _canonical_sha256(
        {
            "role": manifest.role,
            "registry_sha256": manifest.registry_sha256,
            "example_sha256s": manifest.example_sha256s,
        }
    )


def _require_sha256(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in _SHA256_CHARS for char in value)
    ):
        raise ValueError(f"{field_name}_sha256_invalid")
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


def _frozen_float32_array(
    value: Any,
    *,
    shape: tuple[int, ...],
    field_name: str,
) -> np.ndarray:
    try:
        resolved = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name}_invalid") from exc
    if resolved.shape != shape:
        raise ValueError(f"{field_name}_shape_invalid")
    if not bool(np.all(np.isfinite(resolved))):
        raise ValueError(f"{field_name}_nonfinite")
    frozen = resolved.copy()
    frozen.setflags(write=False)
    return frozen


__all__ = [
    "PAPC_MODEL_IDS",
    "ROUTELLM_MF_UPSTREAM_REVISION",
    "TEXTMF_CHECKPOINT_FORMAT",
    "TEXTMF_CHECKPOINT_SELECTION",
    "TEXTMF_BATCH_SIZE",
    "TEXTMF_EARLY_STOPPING",
    "TEXTMF_HIDDEN_SIZE",
    "TEXTMF_INNER_TRAIN_FRACTION",
    "TEXTMF_LEARNING_RATE",
    "TEXTMF_NUM_CLASSES",
    "TEXTMF_NUM_EPOCHS",
    "TEXTMF_OPTIMIZER",
    "TEXTMF_PROMPT_NOISE_ALPHA",
    "TEXTMF_RANDOM_SEED",
    "TEXTMF_SERIALIZER_SHA256",
    "TEXTMF_SERIALIZER_SPEC",
    "TEXTMF_TEXT_DIM",
    "TEXTMF_WEIGHT_DECAY",
    "EmbedPrompt",
    "PairedFitOutcome",
    "RouteLLMMFArtifact",
    "RouteLLMMFConfig",
    "RouteLLMMFDevCalibration",
    "RouteLLMMFDevCandidateReceipt",
    "RouteLLMMFDevExample",
    "RouteLLMMFEmbeddingRequest",
    "RouteLLMMFEmbeddingResult",
    "RouteLLMMFFitCorpus",
    "RouteLLMMFPredictor",
    "TextMFCheckpoint",
    "TextMFNumpyModel",
    "TextMFPreference",
    "TextMFTrainingState",
    "TrainTextMF",
    "bind_routellm_mf_artifact",
    "build_textmf_fit_corpus",
    "freeze_dev_threshold",
    "load_routellm_mf_predictor",
    "train_and_export_textmf_checkpoint",
    "upstream_training_dependency_blockers",
]

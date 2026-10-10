"""Provider-neutral MTRouter adapter pinned to the published implementation.

Adapted for PAPC from MTRouter revision
``ee9e4573a128c3e54aa8c83800318bfb783c4dfe`` (MIT).

The adapter deliberately has no provider fallback. Training examples must come
from per-turn uniform roulette trajectories. Learned inference requires the
exact local Qwen embedding model plus a source-bound single-head checkpoint;
if held-out dev selects the required all-strong candidate, test inference uses
an explicit checkpoint-free native shortcut instead.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import random
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from papc.baselines.router_contract import (
    CHEAP_MODEL,
    MTROUTER,
    STRONG_MODEL,
    HistoryRouterView,
    ModelRole,
    RouterArtifactBinding,
    SplitManifest,
    assert_disjoint_manifests,
)
from papc.baselines.router_policies import (
    RouterDecision,
    decide_mtrouter,
    decide_mtrouter_all_strong,
)

MTROUTER_UPSTREAM_REVISION = "ee9e4573a128c3e54aa8c83800318bfb783c4dfe"
HISTORY_ENCODER_MODEL = "Qwen/Qwen3-Embedding-0.6B"
HISTORY_ENCODER_DIM = 1024
HISTORY_MAX_TOKENS = 8192
HISTORY_MIN_RECENT_TURNS = 1
HISTORY_POOLING = "last_token"
MODEL_META_DIM = 8
MODEL_META_HIDDEN_DIM = 32
MODEL_META_EMBED_DIM = 32
MODEL_RESIDUAL_EMBED_DIM = 16
MODEL_OUTPUT_DIM = 64
MODEL_RESIDUAL_L2_WEIGHT = 0.001
MODEL_KNOWLEDGE_CUTOFF_BASE = "2026-01-01"
Q_HIDDEN_DIMS = (256, 256)
Q_DROPOUT = 0.1
VALUE_LAMBDA = 0.0
ROULETTE_PROPENSITY = 0.5
ROULETTE_POLICY = "per_turn_uniform_two_model_roulette"
TRAJECTORIES_PER_TASK = 3
ROULETTE_SEED_PROTOCOL = (
    "sha256(papc.mtrouter.seed.v1,surface,example_sha256,run_index)"
)
AEC_ENABLED = False
AEC_DISABLED_REASON = "surface_error_rule_annotations_unavailable"
TRAINING_SEED = 42
TRAINING_EPOCHS = 100
TRAINING_BATCH_SIZE = 64
TRAINING_LEARNING_RATE = 1.0e-3
TRAINING_OPTIMIZER = "adamw"
TRAINING_WEIGHT_DECAY = 0.01
TRAINING_GRAD_CLIP = 1.0
TRAINING_SCHEDULER = "cosine"
TRAINING_MIN_LR = 1.0e-5
TRAINING_EARLY_STOP = 3
TRAINING_VAL_RATIO = 0.1
TRAINING_SAVE_EVERY = 5
TRAINING_MAX_PERIODIC_CHECKPOINTS = 5
TRAINING_PRECOMPUTE_BATCH_SIZE = 8
TRAINING_HISTORY_BACKEND = "hf_local_files_only"
TRAINING_CANDIDATE_POLICY = (
    "upstream_best_plus_last_5_periodic_plus_final_deduplicated_by_epoch"
)
TRAINING_MAX_LEARNED_CANDIDATES = (
    TRAINING_MAX_PERIODIC_CHECKPOINTS + 2
)
TRAINING_MAX_DEV_CANDIDATES = TRAINING_MAX_LEARNED_CANDIDATES + 1
TRAINING_SCHEMA_VERSION = "papc_mtrouter_train_export_v1"

MTROUTER_TRAINING_SOURCE_SHA256S = {
    "scripts/train_q_function.py": (
        "85e5e8af3764539a84713baf71d67190f771b4e556735d203cf95219ab496ee8"
    ),
    "src/miniagenticrouter/config/research/training.yaml": (
        "e27e3f0318f60cd106556efb4bb0f8e32a49c32ccb446e65ceb51c5f5e1df399"
    ),
    "src/miniagenticrouter/research/training/dataset.py": (
        "d0ecd30318a6c296e0292bb291776fdbda499e1b8dc7dce0f23c7b26803f00ca"
    ),
    "src/miniagenticrouter/research/training/encoders.py": (
        "b2441fcbf7d71d2aa9a30e417793fc43e3bb40442673de23625cb452fcd18601"
    ),
    "src/miniagenticrouter/research/training/model_encoder.py": (
        "0f109c56dc61d1795f13df452dc46dc4c788573f1b9c2bd3861c264b2c0227f5"
    ),
    "src/miniagenticrouter/research/training/q_network.py": (
        "9d6df9e27e2bbc0f8152048b17436d01f3f05de242940a87d332b866962bc6ad"
    ),
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORMAT_OVERHEAD = 20
_FORMAT_OVERHEAD_PER_MESSAGE = 10


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class MTRouterTokenizer(Protocol):
    """Tokenizer operations used by the pinned history serializer."""

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
    ) -> Sequence[Any]: ...

    def decode(self, tokens: Sequence[Any]) -> str: ...


@dataclass(frozen=True)
class MTRouterPublishedConfig:
    """Paper-default MTRouter representation and value-head configuration."""

    history_encoder_model: str = HISTORY_ENCODER_MODEL
    history_encoder_dim: int = HISTORY_ENCODER_DIM
    history_max_tokens: int = HISTORY_MAX_TOKENS
    history_min_recent_turns: int = HISTORY_MIN_RECENT_TURNS
    history_pooling: str = HISTORY_POOLING
    history_l2_normalize: bool = True
    model_meta_dim: int = MODEL_META_DIM
    model_meta_hidden_dim: int = MODEL_META_HIDDEN_DIM
    model_meta_embed_dim: int = MODEL_META_EMBED_DIM
    model_residual_embed_dim: int = MODEL_RESIDUAL_EMBED_DIM
    model_output_dim: int = MODEL_OUTPUT_DIM
    model_residual_l2_weight: float = MODEL_RESIDUAL_L2_WEIGHT
    model_knowledge_cutoff_base: str = MODEL_KNOWLEDGE_CUTOFF_BASE
    q_hidden_dims: tuple[int, int] = Q_HIDDEN_DIMS
    q_dropout: float = Q_DROPOUT
    q_heads: tuple[str, ...] = ("value",)
    lambda_: float = VALUE_LAMBDA
    selection_policy: str = "greedy_argmax_tie_first"
    model_names: tuple[str, str] = (CHEAP_MODEL, STRONG_MODEL)

    @property
    def sha256(self) -> str:
        return _digest(asdict(self))


PUBLISHED_CONFIG = MTRouterPublishedConfig()
MTROUTER_ALL_STRONG_CONFIG_SHA256 = _digest(
    {
        "method": MTROUTER,
        "mode": "all_strong_fallback",
        "model": STRONG_MODEL,
    }
)
MTROUTER_LEARNED_CHECKPOINT_EXECUTION = "learned_checkpoint"
MTROUTER_ALL_STRONG_EXECUTION = "dev_selected_all_strong_native_shortcut"


@dataclass(frozen=True)
class MTRouterPublishedTrainingConfig:
    """Pinned training defaults from the upstream research configuration."""

    epochs: int = TRAINING_EPOCHS
    batch_size: int = TRAINING_BATCH_SIZE
    learning_rate: float = TRAINING_LEARNING_RATE
    lambda_: float = VALUE_LAMBDA
    cost_loss_weight: float = 0.0
    optimizer: str = TRAINING_OPTIMIZER
    weight_decay: float = TRAINING_WEIGHT_DECAY
    grad_clip: float = TRAINING_GRAD_CLIP
    scheduler: str = TRAINING_SCHEDULER
    min_lr: float = TRAINING_MIN_LR
    early_stop: int = TRAINING_EARLY_STOP
    fit_internal_val_ratio: float = TRAINING_VAL_RATIO
    fit_internal_split_seed: int = TRAINING_SEED
    training_seed: int = TRAINING_SEED
    save_every: int = TRAINING_SAVE_EVERY
    max_periodic_checkpoints: int = TRAINING_MAX_PERIODIC_CHECKPOINTS
    precompute: bool = True
    precompute_batch_size: int = TRAINING_PRECOMPUTE_BATCH_SIZE
    history_backend: str = TRAINING_HISTORY_BACKEND
    history_encoder_frozen: bool = True
    awr_enabled: bool = False
    margin_enabled: bool = False
    aec_enabled: bool = False
    candidate_policy: str = TRAINING_CANDIDATE_POLICY

    @property
    def sha256(self) -> str:
        return _digest(asdict(self))


PUBLISHED_TRAINING_CONFIG = MTRouterPublishedTrainingConfig()


@dataclass(frozen=True)
class MTRouterTrainingReadiness:
    """Provider-zero readiness for the pinned local training tranche."""

    upstream_checkout: str
    observed_upstream_revision: str | None
    source_sha256s: tuple[tuple[str, str], ...]
    torch_available: bool
    transformers_available: bool
    qwen_snapshot_path: str | None
    qwen_source_sha256: str | None
    blockers: tuple[str, ...]
    local_files_only: bool = True
    provider_calls: int = 0

    def __post_init__(self) -> None:
        if self.provider_calls != 0:
            raise ValueError("mtrouter_readiness_provider_calls_must_be_zero")
        if self.local_files_only is not True:
            raise ValueError("mtrouter_readiness_must_be_local_files_only")
        if self.qwen_source_sha256 is not None:
            _required_sha256(
                self.qwen_source_sha256,
                "training_qwen_source_sha256",
            )
        pairs = tuple(self.source_sha256s)
        if any(
            not isinstance(path, str)
            or not path
            or not isinstance(digest, str)
            or not _SHA256_RE.fullmatch(digest)
            for path, digest in pairs
        ):
            raise ValueError("mtrouter_training_source_hashes_invalid")
        object.__setattr__(self, "source_sha256s", pairs)
        object.__setattr__(self, "blockers", tuple(self.blockers))

    @property
    def ready(self) -> bool:
        return not self.blockers

    @property
    def sha256(self) -> str:
        return _digest(asdict(self))


def inspect_mtrouter_training_readiness(
    *,
    upstream_checkout: Path,
    qwen_model_path: Path | None = None,
) -> MTRouterTrainingReadiness:
    """Check pinned sources and a local Qwen snapshot without downloading."""
    checkout = Path(upstream_checkout).resolve()
    blockers: list[str] = []
    observed_revision = _git_revision(checkout)
    if observed_revision != MTROUTER_UPSTREAM_REVISION:
        blockers.append("pinned_upstream_revision_unavailable")

    observed_source_hashes: list[tuple[str, str]] = []
    for relative_path, expected_sha256 in sorted(
        MTROUTER_TRAINING_SOURCE_SHA256S.items()
    ):
        source = checkout / relative_path
        if not source.is_file():
            blockers.append(f"upstream_source_missing:{relative_path}")
            continue
        observed_sha256 = _file_sha256(source)
        observed_source_hashes.append((relative_path, observed_sha256))
        if observed_sha256 != expected_sha256:
            blockers.append(f"upstream_source_hash_mismatch:{relative_path}")

    torch_available = _module_available("torch")
    transformers_available = _module_available("transformers")
    if not torch_available:
        blockers.append("torch_dependency_missing")
    if not transformers_available:
        blockers.append("transformers_dependency_missing")

    snapshot = _resolve_local_qwen_snapshot(qwen_model_path)
    qwen_source_sha256: str | None = None
    if snapshot is None:
        blockers.append("local_qwen_snapshot_missing")
    else:
        qwen_files = _qwen_snapshot_source_files(snapshot)
        if not _qwen_snapshot_has_required_files(qwen_files):
            blockers.append("local_qwen_snapshot_incomplete")
        else:
            qwen_source_sha256 = _source_tree_sha256(snapshot, qwen_files)

    return MTRouterTrainingReadiness(
        upstream_checkout=str(checkout),
        observed_upstream_revision=observed_revision,
        source_sha256s=tuple(observed_source_hashes),
        torch_available=torch_available,
        transformers_available=transformers_available,
        qwen_snapshot_path=str(snapshot) if snapshot is not None else None,
        qwen_source_sha256=qwen_source_sha256,
        blockers=tuple(sorted(set(blockers))),
    )


def _git_revision(checkout: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=checkout,
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, ValueError):
        return None
    revision = result.stdout.strip()
    if result.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", revision):
        return None
    return revision


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _resolve_local_qwen_snapshot(
    qwen_model_path: Path | None,
) -> Path | None:
    if qwen_model_path is not None:
        candidate = Path(qwen_model_path).resolve()
        return candidate if candidate.is_dir() else None
    try:
        huggingface_hub = importlib.import_module("huggingface_hub")
        cached = huggingface_hub.try_to_load_from_cache(
            HISTORY_ENCODER_MODEL,
            "config.json",
        )
    except (ImportError, OSError, ValueError):
        return None
    if not isinstance(cached, str):
        return None
    candidate = Path(cached).resolve().parent
    return candidate if candidate.is_dir() else None


def _qwen_snapshot_source_files(snapshot: Path) -> tuple[Path, ...]:
    selected: list[Path] = []
    for path in snapshot.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if (
            path.suffix in {".json", ".py", ".safetensors", ".bin"}
            or name.startswith(("tokenizer", "vocab", "merges"))
        ):
            selected.append(path)
    return tuple(sorted(selected, key=lambda path: str(path.relative_to(snapshot))))


def _qwen_snapshot_has_required_files(files: Sequence[Path]) -> bool:
    names = {path.name for path in files}
    has_config = "config.json" in names
    has_tokenizer = bool(
        names
        & {
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "vocab.json",
        }
    )
    has_weights = any(
        path.suffix in {".safetensors", ".bin"} for path in files
    )
    return has_config and has_tokenizer and has_weights


def _source_tree_sha256(root: Path, files: Sequence[Path]) -> str:
    return _digest(
        [
            {
                "path": str(path.relative_to(root)),
                "sha256": _file_sha256(path),
            }
            for path in files
        ]
    )

_SERIALIZER_SPEC = {
    "name": "mtrouter_task_recent_history_v1",
    "upstream_revision": MTROUTER_UPSTREAM_REVISION,
    "format": "[Task]\\n{task}\\n\\n[Context]\\n{recent_context}",
    "task_extraction": (
        "task_description_tag",
        "hle_question_block",
        "your_task_is_to_sentence",
        "first_user_message",
    ),
    "context": {
        "skip_system": True,
        "first_user_initial_observation_only": True,
        "newest_messages_first_under_budget": True,
        "format_overhead_per_message": _FORMAT_OVERHEAD_PER_MESSAGE,
    },
    "max_tokens": HISTORY_MAX_TOKENS,
    "min_recent_turns": HISTORY_MIN_RECENT_TURNS,
}
MTROUTER_SERIALIZER_SHA256 = hashlib.sha256(
    json.dumps(
        _SERIALIZER_SPEC,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
).hexdigest()


def mtrouter_messages_from_view(
    view: HistoryRouterView,
) -> tuple[dict[str, str], ...]:
    """Map the method-neutral public view to MTRouter's message serializer."""
    if not isinstance(view, HistoryRouterView):
        raise TypeError("mtrouter_history_view_required")
    messages: list[dict[str, str]] = [
        {
            "role": "user",
            "content": (
                "<task_description>"
                f"{view.original_request}"
                "</task_description>"
            ),
        }
    ]
    for item in view.history_for_runtime():
        role = item.get("role")
        if not isinstance(role, str) or not role.strip():
            role = str(item.get("type", "observation"))
        content = item.get("content")
        if isinstance(content, str) and set(item).issubset({"role", "content"}):
            serialized = content
        else:
            serialized = _canonical_json(item)
        messages.append({"role": role, "content": serialized})
    return tuple(messages)


def segment_mtrouter_history(
    messages: Sequence[Mapping[str, Any]],
    *,
    tokenizer: MTRouterTokenizer,
    max_tokens: int = HISTORY_MAX_TOKENS,
    min_recent_turns: int = HISTORY_MIN_RECENT_TURNS,
) -> str:
    """Serialize task and recent history using the pinned upstream algorithm."""
    if max_tokens != HISTORY_MAX_TOKENS:
        raise ValueError("mtrouter_history_max_tokens_not_frozen")
    if min_recent_turns != HISTORY_MIN_RECENT_TURNS:
        raise ValueError("mtrouter_history_min_recent_turns_not_frozen")
    normalized = _normalize_messages(messages)
    task_segment = _extract_task_description(normalized)
    task_tokens = _count_tokens(task_segment, tokenizer)
    if task_tokens >= max_tokens - _FORMAT_OVERHEAD:
        task_segment = _truncate_to_tokens(
            task_segment,
            max_tokens - _FORMAT_OVERHEAD,
            tokenizer,
        )
        return f"[Task]\n{task_segment}"

    context_messages: list[dict[str, str]] = []
    first_user_seen = False
    for message in normalized:
        role = message["role"]
        if role == "system":
            continue
        if role == "user" and not first_user_seen:
            first_user_seen = True
            observation = _extract_initial_observation(message["content"])
            if observation:
                context_messages.append(
                    {"role": "user", "content": observation}
                )
            continue
        context_messages.append(message)

    remaining = max_tokens - task_tokens - _FORMAT_OVERHEAD
    context_segment = _truncate_recent_context(
        context_messages,
        max_tokens=remaining,
        tokenizer=tokenizer,
    )
    if context_segment:
        return f"[Task]\n{task_segment}\n\n[Context]\n{context_segment}"
    return f"[Task]\n{task_segment}"


@dataclass(frozen=True)
class MTRouterExplorationTurn:
    """One actually visited turn from the uniform two-model roulette policy."""

    example_sha256: str
    episode_sha256: str
    public_input_sha256: str
    turn_index: int
    messages: tuple[dict[str, str], ...] = field(repr=False)
    selection_draw: float
    trajectory_seed: int
    selected_role: ModelRole
    selected_model: str
    propensity: float = ROULETTE_PROPENSITY
    collection_policy: str = ROULETTE_POLICY

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "example_sha256",
            _required_sha256(self.example_sha256, "example_sha256"),
        )
        object.__setattr__(
            self,
            "episode_sha256",
            _required_sha256(self.episode_sha256, "episode_sha256"),
        )
        object.__setattr__(
            self,
            "public_input_sha256",
            _required_sha256(
                self.public_input_sha256,
                "public_input_sha256",
            ),
        )
        if (
            isinstance(self.trajectory_seed, bool)
            or not isinstance(self.trajectory_seed, int)
            or self.trajectory_seed < 0
        ):
            raise ValueError("mtrouter_trajectory_seed_invalid")
        if (
            isinstance(self.turn_index, bool)
            or not isinstance(self.turn_index, int)
            or self.turn_index < 0
        ):
            raise ValueError("mtrouter_turn_index_invalid")
        object.__setattr__(self, "messages", _normalize_messages(self.messages))
        draw = _unit_float(self.selection_draw, "mtrouter_selection_draw")
        if draw == 1.0:
            raise ValueError("mtrouter_selection_draw_must_be_half_open")
        expected_role: ModelRole = "cheap" if draw < 0.5 else "strong"
        if self.selected_role != expected_role:
            raise ValueError("mtrouter_role_not_derived_from_roulette_draw")
        expected_model = (
            CHEAP_MODEL if expected_role == "cheap" else STRONG_MODEL
        )
        if self.selected_model != expected_model:
            raise ValueError("mtrouter_selected_model_role_mismatch")
        if self.propensity != ROULETTE_PROPENSITY:
            raise ValueError("mtrouter_propensity_not_uniform_two_model")
        if self.collection_policy != ROULETTE_POLICY:
            raise ValueError("mtrouter_collection_policy_not_roulette")

    def public_record(self) -> dict[str, Any]:
        return {
            "example_sha256": self.example_sha256,
            "episode_sha256": self.episode_sha256,
            "public_input_sha256": self.public_input_sha256,
            "turn_index": self.turn_index,
            "messages": self.messages,
            "selection_draw": self.selection_draw,
            "trajectory_seed": self.trajectory_seed,
            "selected_role": self.selected_role,
            "selected_model": self.selected_model,
            "propensity": self.propensity,
            "collection_policy": self.collection_policy,
        }


@dataclass(frozen=True)
class MTRouterTrainingRow:
    """One value-head row with the terminal official reward backfilled."""

    exploration: MTRouterExplorationTurn
    terminal_official_reward: float
    target_value: float

    def __post_init__(self) -> None:
        if not isinstance(self.exploration, MTRouterExplorationTurn):
            raise TypeError("mtrouter_exploration_turn_required")
        reward = _unit_float(
            self.terminal_official_reward,
            "mtrouter_terminal_official_reward",
        )
        target = _finite_float(self.target_value, "mtrouter_target_value")
        if target != reward:
            raise ValueError("mtrouter_lambda_zero_target_must_equal_reward")
        object.__setattr__(self, "terminal_official_reward", reward)
        object.__setattr__(self, "target_value", target)

    def public_record(self) -> dict[str, Any]:
        return {
            **self.exploration.public_record(),
            "terminal_official_reward": self.terminal_official_reward,
            "target_value": self.target_value,
            "lambda": VALUE_LAMBDA,
        }


@dataclass(frozen=True)
class MTRouterTrainingTrajectory:
    """A completed roulette trajectory and all of its terminal-value rows."""

    rows: tuple[MTRouterTrainingRow, ...]

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        if not rows:
            raise ValueError("mtrouter_training_trajectory_empty")
        if any(not isinstance(row, MTRouterTrainingRow) for row in rows):
            raise TypeError("mtrouter_training_row_invalid")
        episodes = {row.exploration.episode_sha256 for row in rows}
        if len(episodes) != 1:
            raise ValueError("mtrouter_training_trajectory_episode_mixed")
        examples = {row.exploration.example_sha256 for row in rows}
        if len(examples) != 1:
            raise ValueError("mtrouter_training_trajectory_example_mixed")
        seeds = {row.exploration.trajectory_seed for row in rows}
        if len(seeds) != 1:
            raise ValueError("mtrouter_training_trajectory_seed_mixed")
        indices = tuple(row.exploration.turn_index for row in rows)
        if indices != tuple(range(len(rows))):
            raise ValueError("mtrouter_training_turns_not_contiguous")
        rewards = {row.terminal_official_reward for row in rows}
        if len(rewards) != 1:
            raise ValueError("mtrouter_terminal_reward_not_backfilled")
        object.__setattr__(self, "rows", rows)

    @property
    def episode_sha256(self) -> str:
        return self.rows[0].exploration.episode_sha256

    @property
    def example_sha256(self) -> str:
        return self.rows[0].exploration.example_sha256

    @property
    def trajectory_seed(self) -> int:
        return self.rows[0].exploration.trajectory_seed

    def public_record(self) -> dict[str, Any]:
        return {
            "episode_sha256": self.episode_sha256,
            "example_sha256": self.example_sha256,
            "trajectory_seed": self.trajectory_seed,
            "rows": [row.public_record() for row in self.rows],
        }


@dataclass(frozen=True)
class MTRouterBackendCheckpoint:
    """One file emitted by a local single-head training backend."""

    checkpoint_path: Path
    epoch: int
    origin: str
    train_loss: float
    fit_internal_val_loss: float | None
    checkpoint_metadata: Mapping[str, Any] = field(repr=False)

    def __post_init__(self) -> None:
        path = Path(self.checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError("mtrouter_training_checkpoint_missing")
        object.__setattr__(self, "checkpoint_path", path)
        if (
            isinstance(self.epoch, bool)
            or not isinstance(self.epoch, int)
            or not 1 <= self.epoch <= TRAINING_EPOCHS
        ):
            raise ValueError("mtrouter_training_checkpoint_epoch_invalid")
        if self.origin not in {"fit_internal_best", "periodic", "final"}:
            raise ValueError("mtrouter_training_checkpoint_origin_invalid")
        object.__setattr__(
            self,
            "train_loss",
            _nonnegative_finite_float(
                self.train_loss,
                "mtrouter_training_checkpoint_train_loss",
            ),
        )
        if self.fit_internal_val_loss is not None:
            object.__setattr__(
                self,
                "fit_internal_val_loss",
                _nonnegative_finite_float(
                    self.fit_internal_val_loss,
                    "mtrouter_training_checkpoint_fit_internal_val_loss",
                ),
            )
        if not isinstance(self.checkpoint_metadata, Mapping):
            raise TypeError("mtrouter_training_checkpoint_metadata_invalid")


@dataclass(frozen=True)
class MTRouterTrainingBackendResult:
    """Backend output before source and candidate-policy verification."""

    backend_name: str
    checkpoints: tuple[MTRouterBackendCheckpoint, ...]
    fit_train_row_sha256s: tuple[str, ...]
    fit_internal_val_row_sha256s: tuple[str, ...]
    epochs_completed: int
    qwen_source_sha256: str
    qwen_encoder_frozen: bool
    local_files_only: bool
    provider_calls: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.backend_name, str) or not self.backend_name:
            raise ValueError("mtrouter_training_backend_name_invalid")
        checkpoints = tuple(self.checkpoints)
        if not checkpoints or any(
            not isinstance(item, MTRouterBackendCheckpoint)
            for item in checkpoints
        ):
            raise ValueError("mtrouter_training_backend_checkpoints_invalid")
        object.__setattr__(self, "checkpoints", checkpoints)
        for field_name in (
            "fit_train_row_sha256s",
            "fit_internal_val_row_sha256s",
        ):
            values = tuple(getattr(self, field_name))
            if len(set(values)) != len(values):
                raise ValueError(f"mtrouter_training_{field_name}_duplicate")
            for value in values:
                _required_sha256(value, field_name)
            object.__setattr__(self, field_name, values)
        if (
            isinstance(self.epochs_completed, bool)
            or not isinstance(self.epochs_completed, int)
            or not 1 <= self.epochs_completed <= TRAINING_EPOCHS
        ):
            raise ValueError("mtrouter_training_epochs_completed_invalid")
        _required_sha256(
            self.qwen_source_sha256,
            "training_qwen_source_sha256",
        )
        if self.qwen_encoder_frozen is not True:
            raise ValueError("mtrouter_training_qwen_encoder_not_frozen")
        if self.local_files_only is not True:
            raise ValueError("mtrouter_training_not_local_files_only")
        if self.provider_calls != 0:
            raise ValueError("mtrouter_training_provider_calls_must_be_zero")


class MTRouterTrainingBackend(Protocol):
    """Backend seam used to test orchestration without model downloads."""

    backend_name: str
    qwen_source_sha256: str
    qwen_encoder_frozen: bool
    local_files_only: bool

    def train_and_export(
        self,
        *,
        rows: tuple[MTRouterTrainingRow, ...],
        row_sha256s: tuple[str, ...],
        fit_train_indices: tuple[int, ...],
        fit_internal_val_indices: tuple[int, ...],
        output_dir: Path,
        checkpoint_receipt: Mapping[str, Any],
    ) -> MTRouterTrainingBackendResult: ...


@dataclass(frozen=True)
class MTRouterCheckpointCandidate:
    """A source-bound learned checkpoint eligible for held-out dev replay."""

    checkpoint_path: str
    checkpoint_sha256: str
    epoch: int
    origin: str
    train_loss: float
    fit_internal_val_loss: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint_path, str) or not self.checkpoint_path:
            raise ValueError("mtrouter_candidate_checkpoint_path_invalid")
        _required_sha256(
            self.checkpoint_sha256,
            "candidate_checkpoint_sha256",
        )
        if (
            isinstance(self.epoch, bool)
            or not isinstance(self.epoch, int)
            or not 1 <= self.epoch <= TRAINING_EPOCHS
        ):
            raise ValueError("mtrouter_candidate_epoch_invalid")
        if self.origin not in {"fit_internal_best", "periodic", "final"}:
            raise ValueError("mtrouter_candidate_origin_invalid")
        object.__setattr__(
            self,
            "train_loss",
            _nonnegative_finite_float(
                self.train_loss,
                "mtrouter_candidate_train_loss",
            ),
        )
        if self.fit_internal_val_loss is not None:
            object.__setattr__(
                self,
                "fit_internal_val_loss",
                _nonnegative_finite_float(
                    self.fit_internal_val_loss,
                    "mtrouter_candidate_fit_internal_val_loss",
                ),
            )

    @property
    def candidate_config_sha256(self) -> str:
        return self.checkpoint_sha256

    def public_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MTRouterTrainingExport:
    """Fit-only training receipt plus the official checkpoint family."""

    surface: str
    fit_registry_sha256: str
    fit_membership_sha256: str
    fit_rows_sha256: str
    fit_internal_split_sha256: str
    fit_example_sha256s: tuple[str, ...]
    fit_row_count: int
    fit_train_row_count: int
    fit_internal_val_row_count: int
    epochs_completed: int
    backend_name: str
    qwen_source_sha256: str
    upstream_source_sha256s: tuple[tuple[str, str], ...]
    training_config_sha256: str
    candidates: tuple[MTRouterCheckpointCandidate, ...]
    provider_calls: int = 0
    upstream_revision: str = MTROUTER_UPSTREAM_REVISION
    candidate_policy: str = TRAINING_CANDIDATE_POLICY
    all_strong_config_sha256: str = MTROUTER_ALL_STRONG_CONFIG_SHA256

    def __post_init__(self) -> None:
        if not isinstance(self.surface, str) or not self.surface.strip():
            raise ValueError("mtrouter_training_export_surface_invalid")
        if self.upstream_revision != MTROUTER_UPSTREAM_REVISION:
            raise ValueError("mtrouter_training_export_revision_mismatch")
        if self.candidate_policy != TRAINING_CANDIDATE_POLICY:
            raise ValueError("mtrouter_training_candidate_policy_mismatch")
        if self.all_strong_config_sha256 != (
            MTROUTER_ALL_STRONG_CONFIG_SHA256
        ):
            raise ValueError("mtrouter_training_all_strong_hash_mismatch")
        if self.training_config_sha256 != PUBLISHED_TRAINING_CONFIG.sha256:
            raise ValueError("mtrouter_training_config_hash_mismatch")
        for field_name in (
            "fit_registry_sha256",
            "fit_membership_sha256",
            "fit_rows_sha256",
            "fit_internal_split_sha256",
            "qwen_source_sha256",
        ):
            _required_sha256(getattr(self, field_name), field_name)
        examples = tuple(self.fit_example_sha256s)
        if not examples or len(set(examples)) != len(examples):
            raise ValueError("mtrouter_training_fit_examples_invalid")
        for value in examples:
            _required_sha256(value, "training_fit_example_sha256")
        object.__setattr__(self, "fit_example_sha256s", examples)
        for field_name in (
            "fit_row_count",
            "fit_train_row_count",
            "epochs_completed",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"mtrouter_training_{field_name}_invalid")
        if (
            isinstance(self.fit_internal_val_row_count, bool)
            or not isinstance(self.fit_internal_val_row_count, int)
            or self.fit_internal_val_row_count < 0
        ):
            raise ValueError(
                "mtrouter_training_fit_internal_val_row_count_invalid"
            )
        if (
            self.fit_train_row_count + self.fit_internal_val_row_count
            != self.fit_row_count
        ):
            raise ValueError("mtrouter_training_fit_partition_count_mismatch")
        sources = tuple(self.upstream_source_sha256s)
        if dict(sources) != MTROUTER_TRAINING_SOURCE_SHA256S:
            raise ValueError("mtrouter_training_upstream_sources_mismatch")
        object.__setattr__(self, "upstream_source_sha256s", sources)
        candidates = tuple(self.candidates)
        if (
            not candidates
            or len(candidates) > TRAINING_MAX_LEARNED_CANDIDATES
            or any(
                not isinstance(item, MTRouterCheckpointCandidate)
                for item in candidates
            )
        ):
            raise ValueError("mtrouter_training_candidates_invalid")
        epochs = tuple(item.epoch for item in candidates)
        hashes = tuple(item.checkpoint_sha256 for item in candidates)
        if len(set(epochs)) != len(epochs) or len(set(hashes)) != len(hashes):
            raise ValueError("mtrouter_training_candidates_not_deduplicated")
        object.__setattr__(self, "candidates", candidates)
        if self.provider_calls != 0:
            raise ValueError("mtrouter_training_export_provider_calls_nonzero")

    @property
    def candidate_config_sha256s(self) -> tuple[str, ...]:
        return tuple(
            item.candidate_config_sha256 for item in self.candidates
        )

    @property
    def dev_candidate_config_sha256s(self) -> tuple[str, ...]:
        return (
            *self.candidate_config_sha256s,
            self.all_strong_config_sha256,
        )

    @property
    def sha256(self) -> str:
        return _digest(self.public_record())

    def dev_replay_episode_cap(self, dev_example_count: int) -> int:
        if (
            isinstance(dev_example_count, bool)
            or not isinstance(dev_example_count, int)
            or dev_example_count <= 0
        ):
            raise ValueError("mtrouter_dev_example_count_invalid")
        return len(self.dev_candidate_config_sha256s) * dev_example_count

    def public_record(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "candidates": [
                candidate.public_record() for candidate in self.candidates
            ],
        }

    def json_record(self) -> dict[str, Any]:
        return {
            **self.public_record(),
            "export_sha256": self.sha256,
        }


class MTRouterRouletteCollector:
    """Concrete per-turn 1/2 roulette collector for fit/dev trajectories."""

    def __init__(
        self,
        *,
        surface: str,
        example_sha256: str,
        split: str,
        trajectory_index: int = 0,
    ):
        if not isinstance(surface, str) or not surface.strip():
            raise ValueError("mtrouter_roulette_surface_invalid")
        self.surface = surface.strip()
        self.example_sha256 = _required_sha256(
            example_sha256,
            "example_sha256",
        )
        if split not in {"fit", "dev"}:
            raise ValueError("mtrouter_roulette_split_invalid")
        if (
            isinstance(trajectory_index, bool)
            or not isinstance(trajectory_index, int)
            or not 0 <= trajectory_index < TRAJECTORIES_PER_TASK
        ):
            raise ValueError("mtrouter_trajectory_index_not_frozen")
        seed = _trajectory_seed(
            self.surface,
            self.example_sha256,
            split=split,
            trajectory_index=trajectory_index,
        )
        self.episode_sha256 = _digest(
            {
                "example_sha256": self.example_sha256,
                "surface": self.surface,
                "split": split,
                "trajectory_index": trajectory_index,
                "seed": seed,
            }
        )
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("mtrouter_roulette_seed_must_be_int")
        self._random = random.Random(seed)
        self._seed = seed
        self._turns: list[MTRouterExplorationTurn] = []
        self._finalized = False

    def select(self, view: HistoryRouterView) -> MTRouterExplorationTurn:
        if self._finalized:
            raise RuntimeError("mtrouter_roulette_trajectory_already_finalized")
        if not isinstance(view, HistoryRouterView):
            raise TypeError("mtrouter_history_view_required")
        if view.turn_index != len(self._turns):
            raise ValueError("mtrouter_roulette_turn_index_not_contiguous")
        draw = self._random.random()
        role: ModelRole = "cheap" if draw < 0.5 else "strong"
        turn = MTRouterExplorationTurn(
            example_sha256=self.example_sha256,
            episode_sha256=self.episode_sha256,
            public_input_sha256=view.sha256,
            turn_index=view.turn_index,
            messages=mtrouter_messages_from_view(view),
            selection_draw=draw,
            trajectory_seed=self._seed,
            selected_role=role,
            selected_model=CHEAP_MODEL if role == "cheap" else STRONG_MODEL,
        )
        self._turns.append(turn)
        return turn

    def finalize(
        self,
        *,
        terminal_official_reward: float,
    ) -> MTRouterTrainingTrajectory:
        if self._finalized:
            raise RuntimeError("mtrouter_roulette_trajectory_already_finalized")
        if not self._turns:
            raise ValueError("mtrouter_roulette_no_visited_turns")
        reward = _unit_float(
            terminal_official_reward,
            "mtrouter_terminal_official_reward",
        )
        self._finalized = True
        return MTRouterTrainingTrajectory(
            rows=tuple(
                MTRouterTrainingRow(
                    exploration=turn,
                    terminal_official_reward=reward,
                    target_value=reward,
                )
                for turn in self._turns
            )
        )


@dataclass(frozen=True)
class MTRouterDevEvaluation:
    example_sha256: str
    candidate_config_sha256: str
    official_quality: float
    provider_api_cost_usd: float
    strong_call_count: int
    all_strong: bool = False

    def __post_init__(self) -> None:
        _required_sha256(self.example_sha256, "mtrouter_dev_example")
        _required_sha256(
            self.candidate_config_sha256,
            "mtrouter_dev_candidate_config",
        )
        object.__setattr__(
            self,
            "official_quality",
            _unit_float(self.official_quality, "mtrouter_dev_quality"),
        )
        cost = _finite_float(
            self.provider_api_cost_usd,
            "mtrouter_dev_provider_api_cost",
        )
        if cost < 0.0:
            raise ValueError("mtrouter_dev_provider_api_cost_negative")
        object.__setattr__(self, "provider_api_cost_usd", cost)
        if (
            isinstance(self.strong_call_count, bool)
            or not isinstance(self.strong_call_count, int)
            or self.strong_call_count < 0
        ):
            raise ValueError("mtrouter_dev_strong_call_count_invalid")
        if not isinstance(self.all_strong, bool):
            raise TypeError("mtrouter_dev_all_strong_must_be_bool")
        if (
            self.all_strong
            and self.candidate_config_sha256
            != MTROUTER_ALL_STRONG_CONFIG_SHA256
        ):
            raise ValueError("mtrouter_all_strong_config_digest_mismatch")

    def public_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MTRouterDevCandidateReceipt:
    candidate_config_sha256: str
    dev_quality: float
    provider_api_cost_usd: float
    strong_call_count: int
    all_strong: bool
    example_membership_sha256: str

    def __post_init__(self) -> None:
        for field_name in (
            "candidate_config_sha256",
            "example_membership_sha256",
        ):
            _required_sha256(getattr(self, field_name), field_name)
        object.__setattr__(
            self,
            "dev_quality",
            _unit_float(self.dev_quality, "mtrouter_dev_quality"),
        )
        cost = _finite_float(
            self.provider_api_cost_usd,
            "mtrouter_dev_provider_api_cost",
        )
        if cost < 0.0:
            raise ValueError("mtrouter_dev_provider_api_cost_negative")
        object.__setattr__(self, "provider_api_cost_usd", cost)
        if (
            isinstance(self.strong_call_count, bool)
            or not isinstance(self.strong_call_count, int)
            or self.strong_call_count < 0
        ):
            raise ValueError("mtrouter_dev_strong_call_count_invalid")
        if not isinstance(self.all_strong, bool):
            raise TypeError("mtrouter_dev_all_strong_must_be_bool")

    def public_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MTRouterDevSelection:
    dev_registry_sha256: str
    dev_membership_sha256: str
    dev_rows_sha256: str
    strong_full_dev_anchor: float
    selected_candidate_config_sha256: str
    selected_all_strong: bool
    candidate_receipts: tuple[MTRouterDevCandidateReceipt, ...]
    selection_rule: str = (
        "strong_full_dev_anchor_then_min_provider_cost_then_min_strong_calls_then_config_sha256"
    )

    def __post_init__(self) -> None:
        for field_name in (
            "dev_registry_sha256",
            "dev_membership_sha256",
            "dev_rows_sha256",
            "selected_candidate_config_sha256",
        ):
            _required_sha256(getattr(self, field_name), field_name)
        object.__setattr__(
            self,
            "strong_full_dev_anchor",
            _unit_float(
                self.strong_full_dev_anchor,
                "mtrouter_strong_full_dev_anchor",
            ),
        )
        receipts = tuple(self.candidate_receipts)
        if not receipts or any(
            not isinstance(item, MTRouterDevCandidateReceipt)
            for item in receipts
        ):
            raise ValueError("mtrouter_dev_candidate_receipts_invalid")
        if sum(item.all_strong for item in receipts) != 1:
            raise ValueError("mtrouter_dev_all_strong_candidate_required")
        if not any(
            item.candidate_config_sha256
            == self.selected_candidate_config_sha256
            and item.all_strong == self.selected_all_strong
            for item in receipts
        ):
            raise ValueError("mtrouter_dev_selected_candidate_missing")
        object.__setattr__(self, "candidate_receipts", receipts)
        if self.selection_rule != (
            "strong_full_dev_anchor_then_min_provider_cost_then_min_strong_calls_then_config_sha256"
        ):
            raise ValueError("mtrouter_dev_selection_rule_invalid")

    @property
    def sha256(self) -> str:
        return _digest(self.public_record())

    def public_record(self) -> dict[str, Any]:
        return {
            "dev_registry_sha256": self.dev_registry_sha256,
            "dev_membership_sha256": self.dev_membership_sha256,
            "dev_rows_sha256": self.dev_rows_sha256,
            "strong_full_dev_anchor": self.strong_full_dev_anchor,
            "selected_candidate_config_sha256": (
                self.selected_candidate_config_sha256
            ),
            "selected_all_strong": self.selected_all_strong,
            "candidate_receipts": [
                item.public_record() for item in self.candidate_receipts
            ],
            "selection_rule": self.selection_rule,
        }


def select_mtrouter_dev_candidate(
    evaluations: Sequence[MTRouterDevEvaluation],
    *,
    dev_manifest: SplitManifest,
) -> MTRouterDevSelection:
    if isinstance(evaluations, (str, bytes)) or not isinstance(
        evaluations,
        Sequence,
    ):
        raise TypeError("mtrouter_dev_evaluations_must_be_sequence")
    rows = tuple(evaluations)
    if not rows or any(
        not isinstance(item, MTRouterDevEvaluation) for item in rows
    ):
        raise ValueError("mtrouter_dev_evaluations_invalid")
    if not isinstance(dev_manifest, SplitManifest):
        raise TypeError("mtrouter_dev_manifest_invalid")
    if dev_manifest.role != "router_dev":
        raise ValueError("mtrouter_dev_manifest_role_invalid")
    grouped: dict[str, list[MTRouterDevEvaluation]] = {}
    for row in rows:
        grouped.setdefault(row.candidate_config_sha256, []).append(row)
    receipts: list[MTRouterDevCandidateReceipt] = []
    for candidate_sha256, candidate_rows in sorted(grouped.items()):
        identities = tuple(item.example_sha256 for item in candidate_rows)
        _require_exact_membership(
            dev_manifest,
            identities,
            field_name="mtrouter_dev_candidate",
        )
        all_strong_values = {item.all_strong for item in candidate_rows}
        if len(all_strong_values) != 1:
            raise ValueError("mtrouter_dev_candidate_mode_mixed")
        receipts.append(
            MTRouterDevCandidateReceipt(
                candidate_config_sha256=candidate_sha256,
                dev_quality=sum(
                    item.official_quality for item in candidate_rows
                )
                / len(candidate_rows),
                provider_api_cost_usd=sum(
                    item.provider_api_cost_usd for item in candidate_rows
                ),
                strong_call_count=sum(
                    item.strong_call_count for item in candidate_rows
                ),
                all_strong=next(iter(all_strong_values)),
                example_membership_sha256=_membership_sha256(
                    dev_manifest,
                    identities,
                ),
            )
        )
    all_strong = tuple(item for item in receipts if item.all_strong)
    if len(all_strong) != 1:
        raise ValueError("mtrouter_dev_all_strong_candidate_required")
    anchor = all_strong[0].dev_quality
    feasible = tuple(
        item for item in receipts if item.dev_quality + 1e-12 >= anchor
    )
    if not feasible:
        raise ValueError("mtrouter_no_dev_candidate_meets_strong_anchor")
    selected = min(
        feasible,
        key=lambda item: (
            item.provider_api_cost_usd,
            item.strong_call_count,
            item.candidate_config_sha256,
        ),
    )
    return MTRouterDevSelection(
        dev_registry_sha256=dev_manifest.registry_sha256,
        dev_membership_sha256=_membership_sha256(
            dev_manifest,
            dev_manifest.example_sha256s,
        ),
        dev_rows_sha256=_digest(
            [item.public_record() for item in rows]
        ),
        strong_full_dev_anchor=anchor,
        selected_candidate_config_sha256=(
            selected.candidate_config_sha256
        ),
        selected_all_strong=selected.all_strong,
        candidate_receipts=tuple(receipts),
    )


@dataclass(frozen=True)
class MTRouterDevReplayPlan:
    """Exact held-out dev candidate family, including all-strong fallback."""

    training_export_sha256: str
    dev_registry_sha256: str
    dev_membership_sha256: str
    learned_candidate_config_sha256s: tuple[str, ...]
    all_strong_config_sha256: str
    dev_example_count: int
    dev_candidate_count: int
    dev_replay_episode_cap: int
    maximum_policy_episode_cap: int
    provider_calls_at_plan_time: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "training_export_sha256",
            "dev_registry_sha256",
            "dev_membership_sha256",
        ):
            _required_sha256(getattr(self, field_name), field_name)
        candidates = tuple(self.learned_candidate_config_sha256s)
        if (
            not candidates
            or len(candidates) > TRAINING_MAX_LEARNED_CANDIDATES
            or len(set(candidates)) != len(candidates)
        ):
            raise ValueError("mtrouter_dev_plan_learned_candidates_invalid")
        for candidate in candidates:
            _required_sha256(candidate, "dev_plan_candidate_sha256")
        object.__setattr__(
            self,
            "learned_candidate_config_sha256s",
            candidates,
        )
        if self.all_strong_config_sha256 != (
            MTROUTER_ALL_STRONG_CONFIG_SHA256
        ):
            raise ValueError("mtrouter_dev_plan_all_strong_hash_mismatch")
        if (
            isinstance(self.dev_example_count, bool)
            or not isinstance(self.dev_example_count, int)
            or self.dev_example_count <= 0
        ):
            raise ValueError("mtrouter_dev_plan_example_count_invalid")
        if self.dev_candidate_count != len(candidates) + 1:
            raise ValueError("mtrouter_dev_plan_candidate_count_mismatch")
        if self.dev_replay_episode_cap != (
            self.dev_candidate_count * self.dev_example_count
        ):
            raise ValueError("mtrouter_dev_plan_episode_cap_mismatch")
        if self.maximum_policy_episode_cap != (
            TRAINING_MAX_DEV_CANDIDATES * self.dev_example_count
        ):
            raise ValueError("mtrouter_dev_plan_maximum_cap_mismatch")
        if self.provider_calls_at_plan_time != 0:
            raise ValueError("mtrouter_dev_plan_provider_calls_nonzero")

    @property
    def candidate_config_sha256s(self) -> tuple[str, ...]:
        return (
            *self.learned_candidate_config_sha256s,
            self.all_strong_config_sha256,
        )

    @property
    def sha256(self) -> str:
        return _digest(asdict(self))


def build_mtrouter_dev_replay_plan(
    training_export: MTRouterTrainingExport,
    *,
    dev_manifest: SplitManifest,
) -> MTRouterDevReplayPlan:
    """Bind all official checkpoints and all-strong to unseen router dev."""
    if not isinstance(training_export, MTRouterTrainingExport):
        raise TypeError("mtrouter_training_export_required")
    if not isinstance(dev_manifest, SplitManifest):
        raise TypeError("mtrouter_dev_manifest_invalid")
    if dev_manifest.role != "router_dev":
        raise ValueError("mtrouter_dev_manifest_role_invalid")
    if training_export.fit_registry_sha256 == dev_manifest.registry_sha256:
        raise ValueError("mtrouter_fit_dev_registry_reused")
    if set(training_export.fit_example_sha256s) & set(
        dev_manifest.example_sha256s
    ):
        raise ValueError("mtrouter_fit_dev_example_overlap")
    learned = training_export.candidate_config_sha256s
    dev_example_count = len(dev_manifest.example_sha256s)
    return MTRouterDevReplayPlan(
        training_export_sha256=training_export.sha256,
        dev_registry_sha256=dev_manifest.registry_sha256,
        dev_membership_sha256=_membership_sha256(
            dev_manifest,
            dev_manifest.example_sha256s,
        ),
        learned_candidate_config_sha256s=learned,
        all_strong_config_sha256=MTROUTER_ALL_STRONG_CONFIG_SHA256,
        dev_example_count=dev_example_count,
        dev_candidate_count=len(learned) + 1,
        dev_replay_episode_cap=(len(learned) + 1) * dev_example_count,
        maximum_policy_episode_cap=(
            TRAINING_MAX_DEV_CANDIDATES * dev_example_count
        ),
    )


def select_trained_mtrouter_dev_candidate(
    evaluations: Sequence[MTRouterDevEvaluation],
    *,
    dev_manifest: SplitManifest,
    training_export: MTRouterTrainingExport,
) -> MTRouterDevSelection:
    """Select only after exact held-out replay of the frozen candidate family."""
    plan = build_mtrouter_dev_replay_plan(
        training_export,
        dev_manifest=dev_manifest,
    )
    rows = tuple(evaluations)
    candidate_hashes = {row.candidate_config_sha256 for row in rows}
    if candidate_hashes != set(plan.candidate_config_sha256s):
        raise ValueError("mtrouter_dev_replay_candidate_family_mismatch")
    if len(rows) != plan.dev_replay_episode_cap:
        raise ValueError("mtrouter_dev_replay_row_count_mismatch")
    selection = select_mtrouter_dev_candidate(
        rows,
        dev_manifest=dev_manifest,
    )
    if (
        not selection.selected_all_strong
        and selection.selected_candidate_config_sha256
        not in training_export.candidate_config_sha256s
    ):
        raise ValueError("mtrouter_dev_selected_untrained_checkpoint")
    return selection


@dataclass(frozen=True)
class MTRouterArtifact:
    """Source-bound checkpoint and exact fit/dev substrate for one surface."""

    surface: str
    checkpoint_sha256: str | None
    fit_registry_sha256: str
    dev_registry_sha256: str
    test_registry_sha256: str
    fit_membership_sha256: str
    dev_membership_sha256: str
    split_binding_sha256: str
    fit_rows_sha256: str
    dev_rows_sha256: str
    fit_seed_schedule_sha256: str
    dev_seed_schedule_sha256: str
    fit_trajectory_count: int
    dev_trajectory_count: int
    fit_turn_count: int
    dev_turn_count: int
    trajectories_per_task: int
    dev_selection: MTRouterDevSelection
    execution_mode: str = MTROUTER_LEARNED_CHECKPOINT_EXECUTION
    aec_enabled: bool = AEC_ENABLED
    aec_disabled_reason: str = AEC_DISABLED_REASON
    upstream_revision: str = MTROUTER_UPSTREAM_REVISION
    serializer_sha256: str = MTROUTER_SERIALIZER_SHA256
    config_sha256: str = PUBLISHED_CONFIG.sha256
    model_names: tuple[str, str] = (CHEAP_MODEL, STRONG_MODEL)

    def __post_init__(self) -> None:
        if not isinstance(self.surface, str) or not self.surface.strip():
            raise ValueError("mtrouter_surface_invalid")
        if self.upstream_revision != MTROUTER_UPSTREAM_REVISION:
            raise ValueError("mtrouter_upstream_revision_not_frozen")
        if self.serializer_sha256 != MTROUTER_SERIALIZER_SHA256:
            raise ValueError("mtrouter_serializer_hash_mismatch")
        if self.config_sha256 != PUBLISHED_CONFIG.sha256:
            raise ValueError("mtrouter_config_hash_mismatch")
        model_names = tuple(self.model_names)
        if model_names != (CHEAP_MODEL, STRONG_MODEL):
            raise ValueError("mtrouter_model_order_not_frozen")
        object.__setattr__(self, "model_names", model_names)
        for field_name in (
            "fit_registry_sha256",
            "dev_registry_sha256",
            "test_registry_sha256",
            "fit_membership_sha256",
            "dev_membership_sha256",
            "split_binding_sha256",
            "fit_rows_sha256",
            "dev_rows_sha256",
            "fit_seed_schedule_sha256",
            "dev_seed_schedule_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_sha256(getattr(self, field_name), field_name),
            )
        if self.execution_mode not in {
            MTROUTER_LEARNED_CHECKPOINT_EXECUTION,
            MTROUTER_ALL_STRONG_EXECUTION,
        }:
            raise ValueError("mtrouter_execution_mode_invalid")
        if len(
            {
                self.fit_registry_sha256,
                self.dev_registry_sha256,
                self.test_registry_sha256,
            }
        ) != 3:
            raise ValueError("mtrouter_split_registry_reused")
        if self.fit_rows_sha256 == self.dev_rows_sha256:
            raise ValueError("mtrouter_fit_dev_rows_reused")
        for field_name in (
            "fit_trajectory_count",
            "dev_trajectory_count",
            "fit_turn_count",
            "dev_turn_count",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"mtrouter_{field_name}_invalid")
        if self.fit_turn_count < self.fit_trajectory_count:
            raise ValueError("mtrouter_fit_turn_count_too_small")
        if self.dev_turn_count < self.dev_trajectory_count:
            raise ValueError("mtrouter_dev_turn_count_too_small")
        if self.trajectories_per_task != TRAJECTORIES_PER_TASK:
            raise ValueError("mtrouter_trajectories_per_task_not_frozen")
        if self.aec_enabled is not AEC_ENABLED:
            raise ValueError("mtrouter_aec_must_be_disabled")
        if self.aec_disabled_reason != AEC_DISABLED_REASON:
            raise ValueError("mtrouter_aec_disabled_reason_not_frozen")
        if not isinstance(self.dev_selection, MTRouterDevSelection):
            raise TypeError("mtrouter_dev_selection_invalid")
        if (
            self.dev_selection.dev_registry_sha256
            != self.dev_registry_sha256
            or self.dev_selection.dev_membership_sha256
            != self.dev_membership_sha256
        ):
            raise ValueError("mtrouter_dev_selection_registry_mismatch")
        if not any(
            not receipt.all_strong
            for receipt in self.dev_selection.candidate_receipts
        ):
            raise ValueError(
                "mtrouter_dev_learned_candidate_receipt_required"
            )
        if self.dev_selection.selected_all_strong:
            if self.dev_selection.selected_candidate_config_sha256 != (
                MTROUTER_ALL_STRONG_CONFIG_SHA256
            ):
                raise ValueError("mtrouter_selected_all_strong_hash_mismatch")
            if self.execution_mode != MTROUTER_ALL_STRONG_EXECUTION:
                raise ValueError(
                    "mtrouter_all_strong_execution_mode_required"
                )
            if self.checkpoint_sha256 is not None:
                raise ValueError("mtrouter_all_strong_checkpoint_forbidden")
        else:
            if self.execution_mode != MTROUTER_LEARNED_CHECKPOINT_EXECUTION:
                raise ValueError(
                    "mtrouter_learned_checkpoint_execution_mode_required"
                )
            object.__setattr__(
                self,
                "checkpoint_sha256",
                _required_sha256(
                    self.checkpoint_sha256,
                    "checkpoint_sha256",
                ),
            )
            if (
                self.dev_selection.selected_candidate_config_sha256
                != self.checkpoint_sha256
            ):
                raise ValueError("mtrouter_selected_checkpoint_mismatch")

    @property
    def artifact_sha256(self) -> str:
        return _digest(self.public_record())

    def public_record(self) -> dict[str, Any]:
        return asdict(self)

    def json_record(self) -> dict[str, Any]:
        return {
            **self.public_record(),
            "artifact_sha256": self.artifact_sha256,
        }

    def router_binding(self) -> RouterArtifactBinding:
        return RouterArtifactBinding(
            method=MTROUTER,
            surface=self.surface,
            upstream_revision=self.upstream_revision,
            artifact_sha256=self.artifact_sha256,
            fit_registry_sha256=self.fit_registry_sha256,
            dev_registry_sha256=self.dev_registry_sha256,
            test_registry_sha256=self.test_registry_sha256,
            calibration_sha256=self.dev_selection.sha256,
        )

    @property
    def uses_all_strong_shortcut(self) -> bool:
        return self.execution_mode == MTROUTER_ALL_STRONG_EXECUTION


def build_mtrouter_artifact(
    *,
    surface: str,
    checkpoint_path: Path | None,
    fit_manifest: SplitManifest,
    dev_manifest: SplitManifest,
    test_manifest: SplitManifest,
    fit_trajectories: Sequence[MTRouterTrainingTrajectory],
    dev_trajectories: Sequence[MTRouterTrainingTrajectory],
    dev_selection: MTRouterDevSelection,
) -> MTRouterArtifact:
    """Bind the selected runtime and real roulette fit/dev corpora."""
    checkpoint: Path | None
    checkpoint_sha256: str | None
    execution_mode: str
    if dev_selection.selected_all_strong:
        if checkpoint_path is not None:
            raise ValueError("mtrouter_all_strong_checkpoint_forbidden")
        checkpoint = None
        checkpoint_sha256 = None
        execution_mode = MTROUTER_ALL_STRONG_EXECUTION
    else:
        if checkpoint_path is None:
            raise FileNotFoundError("mtrouter_checkpoint_missing")
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError("mtrouter_checkpoint_missing")
        checkpoint_sha256 = _file_sha256(checkpoint)
        execution_mode = MTROUTER_LEARNED_CHECKPOINT_EXECUTION
    assert_disjoint_manifests(fit_manifest, dev_manifest, test_manifest)
    fit = _validate_corpus(fit_trajectories, split="fit")
    dev = _validate_corpus(dev_trajectories, split="dev")
    fit_episodes = {trajectory.episode_sha256 for trajectory in fit}
    dev_episodes = {trajectory.episode_sha256 for trajectory in dev}
    if fit_episodes & dev_episodes:
        raise ValueError("mtrouter_fit_dev_episode_overlap")
    _validate_seeded_registry_corpus(
        fit,
        manifest=fit_manifest,
        surface=surface,
        split="fit",
    )
    _validate_seeded_registry_corpus(
        dev,
        manifest=dev_manifest,
        surface=surface,
        split="dev",
    )
    if (
        dev_selection.dev_registry_sha256 != dev_manifest.registry_sha256
        or dev_selection.dev_membership_sha256
        != _membership_sha256(dev_manifest, dev_manifest.example_sha256s)
    ):
        raise ValueError("mtrouter_dev_selection_manifest_mismatch")
    return MTRouterArtifact(
        surface=surface,
        checkpoint_sha256=checkpoint_sha256,
        fit_registry_sha256=fit_manifest.registry_sha256,
        dev_registry_sha256=dev_manifest.registry_sha256,
        test_registry_sha256=test_manifest.registry_sha256,
        fit_membership_sha256=_membership_sha256(
            fit_manifest,
            tuple(sorted({item.example_sha256 for item in fit})),
        ),
        dev_membership_sha256=_membership_sha256(
            dev_manifest,
            tuple(sorted({item.example_sha256 for item in dev})),
        ),
        split_binding_sha256=_digest(
            {
                "fit": fit_manifest.__dict__,
                "dev": dev_manifest.__dict__,
                "test": test_manifest.__dict__,
            }
        ),
        fit_rows_sha256=_digest(
            [trajectory.public_record() for trajectory in fit]
        ),
        dev_rows_sha256=_digest(
            [trajectory.public_record() for trajectory in dev]
        ),
        fit_seed_schedule_sha256=_seed_schedule_sha256(
            fit_manifest,
            surface=surface,
            split="fit",
        ),
        dev_seed_schedule_sha256=_seed_schedule_sha256(
            dev_manifest,
            surface=surface,
            split="dev",
        ),
        fit_trajectory_count=len(fit),
        dev_trajectory_count=len(dev),
        fit_turn_count=sum(len(trajectory.rows) for trajectory in fit),
        dev_turn_count=sum(len(trajectory.rows) for trajectory in dev),
        trajectories_per_task=TRAJECTORIES_PER_TASK,
        dev_selection=dev_selection,
        execution_mode=execution_mode,
    )


def load_mtrouter_artifact(path: Path) -> MTRouterArtifact:
    """Load and verify a JSON artifact manifest."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("mtrouter_artifact_record_must_be_mapping")
    claimed_sha256 = raw.pop("artifact_sha256", None)
    selection_record = raw.get("dev_selection")
    if not isinstance(selection_record, dict):
        raise ValueError("mtrouter_artifact_dev_selection_invalid")
    try:
        raw["dev_selection"] = MTRouterDevSelection(
            dev_registry_sha256=selection_record["dev_registry_sha256"],
            dev_membership_sha256=selection_record[
                "dev_membership_sha256"
            ],
            dev_rows_sha256=selection_record["dev_rows_sha256"],
            strong_full_dev_anchor=selection_record[
                "strong_full_dev_anchor"
            ],
            selected_candidate_config_sha256=selection_record[
                "selected_candidate_config_sha256"
            ],
            selected_all_strong=selection_record["selected_all_strong"],
            candidate_receipts=tuple(
                MTRouterDevCandidateReceipt(**item)
                for item in selection_record["candidate_receipts"]
            ),
            selection_rule=selection_record["selection_rule"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("mtrouter_artifact_dev_selection_invalid") from exc
    try:
        artifact = MTRouterArtifact(**raw)
    except TypeError as exc:
        raise ValueError("mtrouter_artifact_schema_invalid") from exc
    if claimed_sha256 != artifact.artifact_sha256:
        raise ValueError("mtrouter_artifact_digest_mismatch")
    return artifact


def load_mtrouter_training_export(path: Path) -> MTRouterTrainingExport:
    """Load a fit-only training export and re-hash every learned checkpoint."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("mtrouter_training_export_record_must_be_mapping")
    claimed_sha256 = raw.pop("export_sha256", None)
    candidates = raw.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("mtrouter_training_export_candidates_invalid")
    try:
        raw["candidates"] = tuple(
            MTRouterCheckpointCandidate(**item) for item in candidates
        )
        raw["fit_example_sha256s"] = tuple(raw["fit_example_sha256s"])
        raw["upstream_source_sha256s"] = tuple(
            tuple(item) for item in raw["upstream_source_sha256s"]
        )
        export = MTRouterTrainingExport(**raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("mtrouter_training_export_schema_invalid") from exc
    if claimed_sha256 != export.sha256:
        raise ValueError("mtrouter_training_export_digest_mismatch")
    for candidate in export.candidates:
        checkpoint = Path(candidate.checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "mtrouter_training_export_checkpoint_missing"
            )
        if _file_sha256(checkpoint) != candidate.checkpoint_sha256:
            raise ValueError(
                "mtrouter_training_export_checkpoint_digest_mismatch"
            )
    return export


def train_export_mtrouter_checkpoints(
    *,
    surface: str,
    fit_manifest: SplitManifest,
    fit_trajectories: Sequence[MTRouterTrainingTrajectory],
    output_dir: Path,
    upstream_checkout: Path,
    qwen_model_path: Path | None = None,
    device: str = "cpu",
) -> MTRouterTrainingExport:
    """Train locally from fit roulette rows and export official candidates."""
    readiness = inspect_mtrouter_training_readiness(
        upstream_checkout=upstream_checkout,
        qwen_model_path=qwen_model_path,
    )
    if not readiness.ready:
        raise RuntimeError(
            "mtrouter_training_not_ready:" + ",".join(readiness.blockers)
        )
    if (
        readiness.qwen_snapshot_path is None
        or readiness.qwen_source_sha256 is None
    ):
        raise RuntimeError("mtrouter_training_local_qwen_not_ready")
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise RuntimeError(
            "mtrouter_local_dependency_missing:torch,transformers"
        ) from exc
    backend = _TorchLocalMTRouterTrainingBackend(
        torch_module=torch,
        transformers_module=transformers,
        qwen_snapshot_path=Path(readiness.qwen_snapshot_path),
        qwen_source_sha256=readiness.qwen_source_sha256,
        device=device,
    )
    return _train_export_mtrouter_with_backend(
        surface=surface,
        fit_manifest=fit_manifest,
        fit_trajectories=fit_trajectories,
        output_dir=output_dir,
        backend=backend,
    )


def _train_export_mtrouter_with_backend(
    *,
    surface: str,
    fit_manifest: SplitManifest,
    fit_trajectories: Sequence[MTRouterTrainingTrajectory],
    output_dir: Path,
    backend: MTRouterTrainingBackend,
) -> MTRouterTrainingExport:
    """Orchestrate fit-only training; injectable only for deterministic tests."""
    if not isinstance(surface, str) or not surface.strip():
        raise ValueError("mtrouter_training_surface_invalid")
    if not isinstance(fit_manifest, SplitManifest):
        raise TypeError("mtrouter_fit_manifest_invalid")
    fit = _validate_corpus(fit_trajectories, split="fit")
    _validate_seeded_registry_corpus(
        fit,
        manifest=fit_manifest,
        surface=surface,
        split="fit",
    )
    rows = tuple(row for trajectory in fit for row in trajectory.rows)
    row_sha256s = tuple(_digest(row.public_record()) for row in rows)
    if len(set(row_sha256s)) != len(row_sha256s):
        raise ValueError("mtrouter_training_fit_row_duplicate")
    train_indices, val_indices = _fit_internal_partition(len(rows))
    split_record = {
        "scope": "router_fit_only",
        "algorithm": "deterministic_shuffle_then_tail_validation",
        "seed": TRAINING_SEED,
        "ratio": TRAINING_VAL_RATIO,
        "fit_train_row_sha256s": [
            row_sha256s[index] for index in train_indices
        ],
        "fit_internal_val_row_sha256s": [
            row_sha256s[index] for index in val_indices
        ],
    }
    fit_rows_sha256 = _digest(
        [trajectory.public_record() for trajectory in fit]
    )
    fit_membership_sha256 = _membership_sha256(
        fit_manifest,
        tuple(sorted({item.example_sha256 for item in fit})),
    )
    qwen_source_sha256 = _required_sha256(
        getattr(backend, "qwen_source_sha256", None),
        "training_qwen_source_sha256",
    )
    if getattr(backend, "qwen_encoder_frozen", None) is not True:
        raise ValueError("mtrouter_training_qwen_encoder_not_frozen")
    if getattr(backend, "local_files_only", None) is not True:
        raise ValueError("mtrouter_training_backend_not_local_files_only")
    backend_name = getattr(backend, "backend_name", None)
    if not isinstance(backend_name, str) or not backend_name:
        raise ValueError("mtrouter_training_backend_name_invalid")
    checkpoint_receipt = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "surface": surface,
        "upstream_revision": MTROUTER_UPSTREAM_REVISION,
        "upstream_source_sha256s": dict(
            sorted(MTROUTER_TRAINING_SOURCE_SHA256S.items())
        ),
        "training_config": asdict(PUBLISHED_TRAINING_CONFIG),
        "training_config_sha256": PUBLISHED_TRAINING_CONFIG.sha256,
        "training_scope": "router_fit_only",
        "fit_registry_sha256": fit_manifest.registry_sha256,
        "fit_membership_sha256": fit_membership_sha256,
        "fit_rows_sha256": fit_rows_sha256,
        "fit_internal_split_sha256": _digest(split_record),
        "qwen_source_sha256": qwen_source_sha256,
        "qwen_encoder_model": HISTORY_ENCODER_MODEL,
        "qwen_encoder_frozen": True,
        "history_backend": TRAINING_HISTORY_BACKEND,
        "local_files_only": True,
        "backend_name": backend_name,
        "provider_calls": 0,
    }
    target = _prepare_training_output_dir(output_dir)
    result = backend.train_and_export(
        rows=rows,
        row_sha256s=row_sha256s,
        fit_train_indices=train_indices,
        fit_internal_val_indices=val_indices,
        output_dir=target,
        checkpoint_receipt=checkpoint_receipt,
    )
    if not isinstance(result, MTRouterTrainingBackendResult):
        raise TypeError("mtrouter_training_backend_result_invalid")
    if (
        result.backend_name != backend_name
        or result.qwen_source_sha256 != qwen_source_sha256
    ):
        raise ValueError("mtrouter_training_backend_provenance_mismatch")
    expected_train_rows = tuple(row_sha256s[index] for index in train_indices)
    expected_val_rows = tuple(row_sha256s[index] for index in val_indices)
    if result.fit_train_row_sha256s != expected_train_rows:
        raise ValueError("mtrouter_training_fit_train_partition_mismatch")
    if result.fit_internal_val_row_sha256s != expected_val_rows:
        raise ValueError("mtrouter_training_fit_val_partition_mismatch")

    snapshots = _validate_backend_checkpoint_family(
        result,
        expected_receipt=checkpoint_receipt,
        has_fit_internal_val=bool(val_indices),
    )
    candidates = _deduplicate_checkpoint_candidates(snapshots)
    export = MTRouterTrainingExport(
        surface=surface,
        fit_registry_sha256=fit_manifest.registry_sha256,
        fit_membership_sha256=fit_membership_sha256,
        fit_rows_sha256=fit_rows_sha256,
        fit_internal_split_sha256=_digest(split_record),
        fit_example_sha256s=fit_manifest.example_sha256s,
        fit_row_count=len(rows),
        fit_train_row_count=len(train_indices),
        fit_internal_val_row_count=len(val_indices),
        epochs_completed=result.epochs_completed,
        backend_name=result.backend_name,
        qwen_source_sha256=result.qwen_source_sha256,
        upstream_source_sha256s=tuple(
            sorted(MTROUTER_TRAINING_SOURCE_SHA256S.items())
        ),
        training_config_sha256=PUBLISHED_TRAINING_CONFIG.sha256,
        candidates=candidates,
    )
    (target / "training_export.json").write_text(
        _canonical_json(export.json_record()) + "\n",
        encoding="utf-8",
    )
    return export


def _prepare_training_output_dir(path: Path) -> Path:
    target = Path(path).resolve()
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("mtrouter_training_output_dir_not_empty")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _fit_internal_partition(
    row_count: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if (
        isinstance(row_count, bool)
        or not isinstance(row_count, int)
        or row_count <= 0
    ):
        raise ValueError("mtrouter_training_row_count_invalid")
    indices = list(range(row_count))
    random.Random(TRAINING_SEED).shuffle(indices)
    val_size = int(row_count * TRAINING_VAL_RATIO)
    train_size = row_count - val_size
    return tuple(indices[:train_size]), tuple(indices[train_size:])


def _validate_backend_checkpoint_family(
    result: MTRouterTrainingBackendResult,
    *,
    expected_receipt: Mapping[str, Any],
    has_fit_internal_val: bool,
) -> tuple[MTRouterBackendCheckpoint, ...]:
    checkpoints = tuple(result.checkpoints)
    finals = tuple(item for item in checkpoints if item.origin == "final")
    best = tuple(
        item for item in checkpoints if item.origin == "fit_internal_best"
    )
    periodic = tuple(
        item for item in checkpoints if item.origin == "periodic"
    )
    if len(finals) != 1 or finals[0].epoch != result.epochs_completed:
        raise ValueError("mtrouter_training_final_checkpoint_invalid")
    if has_fit_internal_val and len(best) != 1:
        raise ValueError("mtrouter_training_best_checkpoint_required")
    if not has_fit_internal_val and best:
        raise ValueError("mtrouter_training_best_without_fit_validation")
    if len(periodic) > TRAINING_MAX_PERIODIC_CHECKPOINTS:
        raise ValueError("mtrouter_training_too_many_periodic_checkpoints")
    if any(
        item.epoch % TRAINING_SAVE_EVERY != 0 for item in periodic
    ):
        raise ValueError("mtrouter_training_periodic_epoch_invalid")
    if tuple(item.epoch for item in periodic) != tuple(
        sorted(item.epoch for item in periodic)
    ):
        raise ValueError("mtrouter_training_periodic_order_invalid")
    for checkpoint in checkpoints:
        validate_mtrouter_checkpoint_metadata(
            checkpoint.checkpoint_metadata
        )
        _validate_training_checkpoint_receipt(
            checkpoint.checkpoint_metadata,
            expected_receipt=expected_receipt,
            epoch=checkpoint.epoch,
            origin=checkpoint.origin,
            train_loss=checkpoint.train_loss,
            fit_internal_val_loss=checkpoint.fit_internal_val_loss,
        )
    return checkpoints


def _validate_training_checkpoint_receipt(
    checkpoint: Mapping[str, Any],
    *,
    expected_receipt: Mapping[str, Any],
    epoch: int,
    origin: str,
    train_loss: float,
    fit_internal_val_loss: float | None,
) -> None:
    receipt = checkpoint.get("papc_training")
    if not isinstance(receipt, Mapping):
        raise ValueError("mtrouter_checkpoint_training_receipt_missing")
    for key, expected in expected_receipt.items():
        if receipt.get(key) != expected:
            raise ValueError(
                f"mtrouter_checkpoint_training_receipt_mismatch:{key}"
            )
    if receipt.get("epoch") != epoch or receipt.get("origin") != origin:
        raise ValueError("mtrouter_checkpoint_epoch_origin_mismatch")
    if receipt.get("train_loss") != train_loss:
        raise ValueError("mtrouter_checkpoint_train_loss_mismatch")
    if receipt.get("fit_internal_val_loss") != fit_internal_val_loss:
        raise ValueError("mtrouter_checkpoint_fit_val_loss_mismatch")


def _deduplicate_checkpoint_candidates(
    checkpoints: Sequence[MTRouterBackendCheckpoint],
) -> tuple[MTRouterCheckpointCandidate, ...]:
    priority = {
        "fit_internal_best": 0,
        "periodic": 1,
        "final": 2,
    }
    by_epoch: dict[int, MTRouterBackendCheckpoint] = {}
    for checkpoint in checkpoints:
        previous = by_epoch.get(checkpoint.epoch)
        if previous is None or priority[checkpoint.origin] < priority[
            previous.origin
        ]:
            by_epoch[checkpoint.epoch] = checkpoint
    candidates = tuple(
        MTRouterCheckpointCandidate(
            checkpoint_path=str(checkpoint.checkpoint_path.resolve()),
            checkpoint_sha256=_file_sha256(checkpoint.checkpoint_path),
            epoch=checkpoint.epoch,
            origin=checkpoint.origin,
            train_loss=checkpoint.train_loss,
            fit_internal_val_loss=checkpoint.fit_internal_val_loss,
        )
        for checkpoint in sorted(
            by_epoch.values(),
            key=lambda item: (item.epoch, priority[item.origin]),
        )
    )
    if len(candidates) > TRAINING_MAX_LEARNED_CANDIDATES:
        raise ValueError("mtrouter_training_candidate_cap_exceeded")
    return candidates


class _TorchLocalMTRouterTrainingBackend:
    """Published single-head optimizer over frozen local Qwen embeddings."""

    backend_name = "torch_local_qwen_single_head"
    qwen_encoder_frozen = True
    local_files_only = True

    def __init__(
        self,
        *,
        torch_module: Any,
        transformers_module: Any,
        qwen_snapshot_path: Path,
        qwen_source_sha256: str,
        device: str,
    ):
        self._torch = torch_module
        self._transformers = transformers_module
        self._qwen_snapshot_path = Path(qwen_snapshot_path).resolve()
        self.qwen_source_sha256 = _required_sha256(
            qwen_source_sha256,
            "training_qwen_source_sha256",
        )
        if not isinstance(device, str) or not device:
            raise ValueError("mtrouter_training_device_invalid")
        self._device = device

    def train_and_export(
        self,
        *,
        rows: tuple[MTRouterTrainingRow, ...],
        row_sha256s: tuple[str, ...],
        fit_train_indices: tuple[int, ...],
        fit_internal_val_indices: tuple[int, ...],
        output_dir: Path,
        checkpoint_receipt: Mapping[str, Any],
    ) -> MTRouterTrainingBackendResult:
        torch = self._torch
        _configure_deterministic_torch(torch)
        tokenizer, history_encoder = self._load_frozen_history_encoder()
        embeddings = self._precompute_embeddings(
            rows,
            tokenizer=tokenizer,
            history_encoder=history_encoder,
        )
        model_encoder, q_network = _build_torch_modules(torch)
        model_encoder = model_encoder.to(self._device)
        q_network = q_network.to(self._device)
        optimizer = _build_published_optimizer(
            torch,
            model_encoder=model_encoder,
            q_network=q_network,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=TRAINING_EPOCHS,
            eta_min=TRAINING_MIN_LR,
        )
        model_indices = torch.tensor(
            [
                0
                if row.exploration.selected_model == CHEAP_MODEL
                else 1
                for row in rows
            ],
            dtype=torch.long,
        )
        targets = torch.tensor(
            [row.target_value for row in rows],
            dtype=torch.float32,
        )
        trainable_parameters = tuple(
            parameter
            for module in (model_encoder, q_network)
            for parameter in module.parameters()
            if parameter.requires_grad
        )

        best_checkpoint: MTRouterBackendCheckpoint | None = None
        periodic_checkpoints: list[MTRouterBackendCheckpoint] = []
        best_val_loss = float("inf")
        epochs_no_improve = 0
        final_train_loss = 0.0
        final_val_loss: float | None = None
        epochs_completed = 0
        for epoch_index in range(TRAINING_EPOCHS):
            epoch = epoch_index + 1
            model_encoder.train()
            q_network.train()
            batch_indices = list(fit_train_indices)
            random.Random(TRAINING_SEED + epoch_index).shuffle(batch_indices)
            batch_losses: list[float] = []
            for start in range(0, len(batch_indices), TRAINING_BATCH_SIZE):
                selected = batch_indices[
                    start : start + TRAINING_BATCH_SIZE
                ]
                selected_tensor = torch.tensor(
                    selected,
                    dtype=torch.long,
                )
                state = embeddings[selected_tensor].to(self._device)
                action = model_indices[selected_tensor].to(self._device)
                target = targets[selected_tensor].to(self._device)
                model_embedding = model_encoder.forward_indices(action)
                prediction = q_network(
                    state,
                    model_embedding,
                ).squeeze(-1)
                score_loss = torch.nn.functional.mse_loss(
                    prediction,
                    target,
                )
                residual_loss = (
                    MODEL_RESIDUAL_L2_WEIGHT
                    * model_encoder.residual_embed.weight.pow(2).sum()
                )
                optimizer.zero_grad()
                (score_loss + residual_loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    max_norm=TRAINING_GRAD_CLIP,
                )
                optimizer.step()
                batch_losses.append(float(score_loss.detach().cpu().item()))
            final_train_loss = sum(batch_losses) / len(batch_losses)
            final_val_loss = _evaluate_fit_internal_loss(
                torch,
                embeddings=embeddings,
                model_indices=model_indices,
                targets=targets,
                indices=fit_internal_val_indices,
                model_encoder=model_encoder,
                q_network=q_network,
                device=self._device,
            )
            epochs_completed = epoch
            if (
                final_val_loss is not None
                and final_val_loss < best_val_loss
            ):
                best_val_loss = final_val_loss
                epochs_no_improve = 0
                best_checkpoint = self._save_checkpoint(
                    output_dir / "q_function_best.pt",
                    model_encoder=model_encoder,
                    q_network=q_network,
                    checkpoint_receipt=checkpoint_receipt,
                    epoch=epoch,
                    origin="fit_internal_best",
                    train_loss=final_train_loss,
                    fit_internal_val_loss=final_val_loss,
                )
            elif final_val_loss is not None:
                epochs_no_improve += 1

            scheduler.step()
            if epoch % TRAINING_SAVE_EVERY == 0:
                periodic_dir = output_dir / "checkpoints"
                periodic_dir.mkdir(parents=True, exist_ok=True)
                periodic_checkpoints.append(
                    self._save_checkpoint(
                        periodic_dir / f"epoch_{epoch:04d}.pt",
                        model_encoder=model_encoder,
                        q_network=q_network,
                        checkpoint_receipt=checkpoint_receipt,
                        epoch=epoch,
                        origin="periodic",
                        train_loss=final_train_loss,
                        fit_internal_val_loss=final_val_loss,
                    )
                )
                while (
                    len(periodic_checkpoints)
                    > TRAINING_MAX_PERIODIC_CHECKPOINTS
                ):
                    removed = periodic_checkpoints.pop(0)
                    removed.checkpoint_path.unlink()
            if (
                final_val_loss is not None
                and epochs_no_improve >= TRAINING_EARLY_STOP
            ):
                break

        final_checkpoint = self._save_checkpoint(
            output_dir / "q_function.pt",
            model_encoder=model_encoder,
            q_network=q_network,
            checkpoint_receipt=checkpoint_receipt,
            epoch=epochs_completed,
            origin="final",
            train_loss=final_train_loss,
            fit_internal_val_loss=final_val_loss,
        )
        checkpoints: list[MTRouterBackendCheckpoint] = []
        if best_checkpoint is not None:
            checkpoints.append(best_checkpoint)
        checkpoints.extend(periodic_checkpoints)
        checkpoints.append(final_checkpoint)
        return MTRouterTrainingBackendResult(
            backend_name=self.backend_name,
            checkpoints=tuple(checkpoints),
            fit_train_row_sha256s=tuple(
                row_sha256s[index] for index in fit_train_indices
            ),
            fit_internal_val_row_sha256s=tuple(
                row_sha256s[index] for index in fit_internal_val_indices
            ),
            epochs_completed=epochs_completed,
            qwen_source_sha256=self.qwen_source_sha256,
            qwen_encoder_frozen=True,
            local_files_only=True,
        )

    def _load_frozen_history_encoder(self) -> tuple[Any, Any]:
        try:
            tokenizer = self._transformers.AutoTokenizer.from_pretrained(
                str(self._qwen_snapshot_path),
                trust_remote_code=True,
                padding_side="left",
                local_files_only=True,
            )
            encoder = self._transformers.AutoModel.from_pretrained(
                str(self._qwen_snapshot_path),
                trust_remote_code=True,
                low_cpu_mem_usage=False,
                local_files_only=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "mtrouter_local_qwen_model_unavailable"
            ) from exc
        if getattr(encoder.config, "hidden_size", None) != HISTORY_ENCODER_DIM:
            raise ValueError("mtrouter_history_encoder_dimension_mismatch")
        for parameter in encoder.parameters():
            parameter.requires_grad = False
        encoder = encoder.to(self._device).eval()
        if any(parameter.requires_grad for parameter in encoder.parameters()):
            raise ValueError("mtrouter_training_qwen_encoder_not_frozen")
        return tokenizer, encoder

    def _precompute_embeddings(
        self,
        rows: tuple[MTRouterTrainingRow, ...],
        *,
        tokenizer: Any,
        history_encoder: Any,
    ) -> Any:
        torch = self._torch
        chunks: list[Any] = []
        for start in range(0, len(rows), TRAINING_PRECOMPUTE_BATCH_SIZE):
            batch = rows[start : start + TRAINING_PRECOMPUTE_BATCH_SIZE]
            texts = [
                segment_mtrouter_history(
                    row.exploration.messages,
                    tokenizer=tokenizer,
                )
                for row in batch
            ]
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=HISTORY_MAX_TOKENS,
                padding=True,
            )
            inputs = {
                key: value.to(self._device)
                for key, value in inputs.items()
            }
            with torch.no_grad():
                hidden = history_encoder(**inputs).last_hidden_state
                attention = inputs["attention_mask"]
                if attention[:, -1].sum() == attention.shape[0]:
                    pooled = hidden[:, -1]
                else:
                    lengths = attention.sum(dim=1) - 1
                    pooled = hidden[
                        torch.arange(
                            hidden.shape[0],
                            device=hidden.device,
                        ),
                        lengths,
                    ]
                pooled = torch.nn.functional.normalize(
                    pooled,
                    p=2,
                    dim=-1,
                )
            chunks.append(pooled.detach().cpu())
        result = torch.cat(chunks, dim=0)
        if tuple(result.shape) != (len(rows), HISTORY_ENCODER_DIM):
            raise ValueError("mtrouter_precomputed_embedding_shape_mismatch")
        return result

    def _save_checkpoint(
        self,
        path: Path,
        *,
        model_encoder: Any,
        q_network: Any,
        checkpoint_receipt: Mapping[str, Any],
        epoch: int,
        origin: str,
        train_loss: float,
        fit_internal_val_loss: float | None,
    ) -> MTRouterBackendCheckpoint:
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = _mtrouter_checkpoint_metadata(
            model_encoder=model_encoder,
            q_network=q_network,
            checkpoint_receipt=checkpoint_receipt,
            epoch=epoch,
            origin=origin,
            train_loss=train_loss,
            fit_internal_val_loss=fit_internal_val_loss,
        )
        self._torch.save(metadata, path)
        return MTRouterBackendCheckpoint(
            checkpoint_path=path,
            epoch=epoch,
            origin=origin,
            train_loss=train_loss,
            fit_internal_val_loss=fit_internal_val_loss,
            checkpoint_metadata=metadata,
        )


def _configure_deterministic_torch(torch: Any) -> None:
    torch.manual_seed(TRAINING_SEED)
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.manual_seed_all(TRAINING_SEED)
    deterministic = getattr(torch, "use_deterministic_algorithms", None)
    if callable(deterministic):
        deterministic(True)
    cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
    if cudnn is not None:
        cudnn.deterministic = True
        cudnn.benchmark = False


def _build_published_optimizer(
    torch: Any,
    *,
    model_encoder: Any,
    q_network: Any,
) -> Any:
    decay_parameters: list[Any] = []
    no_decay_parameters: list[Any] = []
    for prefix, module in (
        ("model_encoder", model_encoder),
        ("q_network", q_network),
    ):
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            qualified_name = f"{prefix}.{name}"
            if "bias" in qualified_name or "norm" in qualified_name:
                no_decay_parameters.append(parameter)
            else:
                decay_parameters.append(parameter)
    return torch.optim.AdamW(
        [
            {
                "params": decay_parameters,
                "weight_decay": TRAINING_WEIGHT_DECAY,
            },
            {
                "params": no_decay_parameters,
                "weight_decay": 0.0,
            },
        ],
        lr=TRAINING_LEARNING_RATE,
    )


def _evaluate_fit_internal_loss(
    torch: Any,
    *,
    embeddings: Any,
    model_indices: Any,
    targets: Any,
    indices: tuple[int, ...],
    model_encoder: Any,
    q_network: Any,
    device: str,
) -> float | None:
    if not indices:
        return None
    model_encoder.eval()
    q_network.eval()
    losses: list[float] = []
    with torch.no_grad():
        for start in range(0, len(indices), TRAINING_BATCH_SIZE):
            selected = indices[start : start + TRAINING_BATCH_SIZE]
            selected_tensor = torch.tensor(selected, dtype=torch.long)
            state = embeddings[selected_tensor].to(device)
            action = model_indices[selected_tensor].to(device)
            target = targets[selected_tensor].to(device)
            prediction = q_network(
                state,
                model_encoder.forward_indices(action),
            ).squeeze(-1)
            loss = torch.nn.functional.mse_loss(prediction, target)
            losses.append(float(loss.detach().cpu().item()))
    return sum(losses) / len(losses)


def _mtrouter_checkpoint_metadata(
    *,
    model_encoder: Any,
    q_network: Any,
    checkpoint_receipt: Mapping[str, Any],
    epoch: int,
    origin: str,
    train_loss: float,
    fit_internal_val_loss: float | None,
) -> dict[str, Any]:
    state_dict = {
        **{
            f"model_encoder.{key}": value.detach().cpu()
            for key, value in model_encoder.state_dict().items()
        },
        **{
            f"q_network.{key}": value.detach().cpu()
            for key, value in q_network.state_dict().items()
        },
    }
    return {
        "state_dict": state_dict,
        "model_names": list(PUBLISHED_CONFIG.model_names),
        "config": _checkpoint_config_record(),
        "lambda_": VALUE_LAMBDA,
        "papc_training": {
            **dict(checkpoint_receipt),
            "epoch": epoch,
            "origin": origin,
            "train_loss": train_loss,
            "fit_internal_val_loss": fit_internal_val_loss,
        },
    }


def _checkpoint_config_record() -> dict[str, Any]:
    return {
        "history_encoder_config": {
            "model_name": HISTORY_ENCODER_MODEL,
            "freeze_encoder": True,
            "max_tokens": HISTORY_MAX_TOKENS,
            "min_recent_turns": HISTORY_MIN_RECENT_TURNS,
            "use_projection": True,
            "pooling_mode": HISTORY_POOLING,
            "backend": "hf",
        },
        "model_encoder_config": {
            "meta_dim": MODEL_META_DIM,
            "meta_hidden_dim": MODEL_META_HIDDEN_DIM,
            "meta_embed_dim": MODEL_META_EMBED_DIM,
            "residual_embed_dim": MODEL_RESIDUAL_EMBED_DIM,
            "output_dim": MODEL_OUTPUT_DIM,
            "residual_l2_weight": MODEL_RESIDUAL_L2_WEIGHT,
            "knowledge_cutoff_base": MODEL_KNOWLEDGE_CUTOFF_BASE,
        },
        "q_network_config": {
            "hidden_dims": list(Q_HIDDEN_DIMS),
            "dropout": Q_DROPOUT,
        },
        "freeze_hf_encoder": True,
    }


@dataclass(frozen=True)
class MTRouterPrediction:
    cheap_value: float
    strong_value: float
    selected_role: ModelRole

    def __post_init__(self) -> None:
        cheap = _finite_float(self.cheap_value, "mtrouter_cheap_value")
        strong = _finite_float(self.strong_value, "mtrouter_strong_value")
        expected: ModelRole = "cheap" if cheap >= strong else "strong"
        if self.selected_role != expected:
            raise ValueError("mtrouter_prediction_not_greedy_argmax")


class MTRouterLocalRuntime:
    """Concrete local Qwen + checkpoint inference; no callback or provider path."""

    def __init__(
        self,
        *,
        artifact: MTRouterArtifact,
        torch_module: Any,
        tokenizer: Any,
        history_encoder: Any,
        model_encoder: Any,
        q_network: Any,
        device: str,
    ):
        self.artifact = artifact
        self._torch = torch_module
        self._tokenizer = tokenizer
        self._history_encoder = history_encoder
        self._model_encoder = model_encoder
        self._q_network = q_network
        self._device = device

    def predict(self, view: HistoryRouterView) -> MTRouterPrediction:
        if not isinstance(view, HistoryRouterView):
            raise TypeError("mtrouter_history_view_required")
        torch = self._torch
        text = segment_mtrouter_history(
            mtrouter_messages_from_view(view),
            tokenizer=self._tokenizer,
        )
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=HISTORY_MAX_TOKENS,
            padding=True,
        )
        inputs = {name: value.to(self._device) for name, value in inputs.items()}
        with torch.no_grad():
            outputs = self._history_encoder(**inputs)
            hidden = outputs.last_hidden_state
            attention = inputs["attention_mask"]
            if attention[:, -1].sum() == attention.shape[0]:
                state = hidden[:, -1]
            else:
                lengths = attention.sum(dim=1) - 1
                state = hidden[
                    torch.arange(hidden.shape[0], device=hidden.device),
                    lengths,
                ]
            state = torch.nn.functional.normalize(state, p=2, dim=-1)
            model_embeddings = self._model_encoder.forward_all()
            expanded = state.expand(len(PUBLISHED_CONFIG.model_names), -1)
            values = self._q_network(expanded, model_embeddings).squeeze(-1)
        cheap_value, strong_value = (
            float(value) for value in values.detach().cpu().tolist()
        )
        role: ModelRole = (
            "cheap" if cheap_value >= strong_value else "strong"
        )
        return MTRouterPrediction(
            cheap_value=cheap_value,
            strong_value=strong_value,
            selected_role=role,
        )

    def decide(
        self,
        *,
        binding: RouterArtifactBinding,
        view: HistoryRouterView,
    ) -> RouterDecision:
        if not isinstance(binding, RouterArtifactBinding):
            raise TypeError("router_artifact_binding_required")
        if binding.method != MTROUTER:
            raise ValueError("mtrouter_binding_method_mismatch")
        if binding.artifact_sha256 != self.artifact.artifact_sha256:
            raise ValueError("mtrouter_binding_artifact_mismatch")
        prediction = self.predict(view)
        return decide_mtrouter(
            binding=binding,
            view=view,
            cheap_predicted_terminal_value=prediction.cheap_value,
            strong_predicted_terminal_value=prediction.strong_value,
        )


@dataclass(frozen=True)
class MTRouterAllStrongRuntime:
    """Checkpoint-free runtime for the held-out-dev all-strong winner."""

    artifact: MTRouterArtifact

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, MTRouterArtifact):
            raise TypeError("mtrouter_artifact_required")
        if not self.artifact.uses_all_strong_shortcut:
            raise ValueError("mtrouter_all_strong_artifact_required")

    def decide(
        self,
        *,
        binding: RouterArtifactBinding,
        view: HistoryRouterView,
    ) -> RouterDecision:
        if not isinstance(binding, RouterArtifactBinding):
            raise TypeError("router_artifact_binding_required")
        if binding.method != MTROUTER:
            raise ValueError("mtrouter_binding_method_mismatch")
        if binding.artifact_sha256 != self.artifact.artifact_sha256:
            raise ValueError("mtrouter_binding_artifact_mismatch")
        return decide_mtrouter_all_strong(binding=binding, view=view)


def load_mtrouter_runtime(
    *,
    artifact: MTRouterArtifact,
    checkpoint_path: Path | None,
    qwen_model_path: Path | None = None,
    device: str = "cpu",
) -> MTRouterLocalRuntime | MTRouterAllStrongRuntime:
    """Load the selected learned or all-strong runtime or fail closed."""
    if not isinstance(artifact, MTRouterArtifact):
        raise TypeError("mtrouter_artifact_required")
    if artifact.uses_all_strong_shortcut:
        if checkpoint_path is not None:
            raise ValueError("mtrouter_all_strong_checkpoint_forbidden")
        return MTRouterAllStrongRuntime(artifact=artifact)
    if checkpoint_path is None:
        raise FileNotFoundError("mtrouter_checkpoint_missing")
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError("mtrouter_checkpoint_missing")
    if _file_sha256(checkpoint_path) != artifact.checkpoint_sha256:
        raise ValueError("mtrouter_checkpoint_digest_mismatch")
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise RuntimeError(
            "mtrouter_local_dependency_missing:torch,transformers"
        ) from exc

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    validate_mtrouter_checkpoint_metadata(checkpoint)

    encoder_source = (
        HISTORY_ENCODER_MODEL
        if qwen_model_path is None
        else str(Path(qwen_model_path).resolve())
    )
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            encoder_source,
            trust_remote_code=True,
            padding_side="left",
            local_files_only=True,
        )
        history_encoder = transformers.AutoModel.from_pretrained(
            encoder_source,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError("mtrouter_local_qwen_model_unavailable") from exc
    if getattr(history_encoder.config, "hidden_size", None) != HISTORY_ENCODER_DIM:
        raise ValueError("mtrouter_history_encoder_dimension_mismatch")

    model_encoder, q_network = _build_torch_modules(torch)
    state_dict = checkpoint["state_dict"]
    model_state = _strip_state_prefix(state_dict, "model_encoder.")
    q_state = _strip_state_prefix(state_dict, "q_network.")
    unexpected = sorted(
        key
        for key in state_dict
        if not key.startswith(("model_encoder.", "q_network."))
    )
    if unexpected:
        raise ValueError("mtrouter_checkpoint_unexpected_state")
    try:
        model_encoder.load_state_dict(model_state, strict=True)
        q_network.load_state_dict(q_state, strict=True)
    except Exception as exc:
        raise ValueError("mtrouter_checkpoint_shape_or_key_mismatch") from exc

    history_encoder = history_encoder.to(device).eval()
    model_encoder = model_encoder.to(device).eval()
    q_network = q_network.to(device).eval()
    return MTRouterLocalRuntime(
        artifact=artifact,
        torch_module=torch,
        tokenizer=tokenizer,
        history_encoder=history_encoder,
        model_encoder=model_encoder,
        q_network=q_network,
        device=device,
    )


def validate_mtrouter_checkpoint_metadata(checkpoint: Any) -> None:
    """Validate the saved upstream config before constructing any network."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("mtrouter_checkpoint_mapping_required")
    if tuple(checkpoint.get("model_names", ())) != (
        CHEAP_MODEL,
        STRONG_MODEL,
    ):
        raise ValueError("mtrouter_checkpoint_model_order_mismatch")
    lambda_ = checkpoint.get("lambda_")
    if (
        isinstance(lambda_, bool)
        or not isinstance(lambda_, (int, float))
        or float(lambda_) != VALUE_LAMBDA
    ):
        raise ValueError("mtrouter_checkpoint_lambda_not_zero")
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("mtrouter_checkpoint_config_missing")
    history = config.get("history_encoder_config")
    model = config.get("model_encoder_config")
    q_config = config.get("q_network_config")
    if not all(isinstance(item, Mapping) for item in (history, model, q_config)):
        raise ValueError("mtrouter_checkpoint_nested_config_missing")
    expected_history = {
        "model_name": HISTORY_ENCODER_MODEL,
        "freeze_encoder": True,
        "max_tokens": HISTORY_MAX_TOKENS,
        "min_recent_turns": HISTORY_MIN_RECENT_TURNS,
        "use_projection": True,
        "pooling_mode": HISTORY_POOLING,
        "backend": "hf",
    }
    if any(history.get(key) != value for key, value in expected_history.items()):
        raise ValueError("mtrouter_checkpoint_history_config_mismatch")
    if config.get("freeze_hf_encoder") is not True:
        raise ValueError("mtrouter_checkpoint_history_not_frozen")
    expected_model = {
        "meta_dim": MODEL_META_DIM,
        "meta_hidden_dim": MODEL_META_HIDDEN_DIM,
        "meta_embed_dim": MODEL_META_EMBED_DIM,
        "residual_embed_dim": MODEL_RESIDUAL_EMBED_DIM,
        "output_dim": MODEL_OUTPUT_DIM,
        "residual_l2_weight": MODEL_RESIDUAL_L2_WEIGHT,
        "knowledge_cutoff_base": MODEL_KNOWLEDGE_CUTOFF_BASE,
    }
    if any(model.get(key) != value for key, value in expected_model.items()):
        raise ValueError("mtrouter_checkpoint_model_config_mismatch")
    if tuple(q_config.get("hidden_dims", ())) != Q_HIDDEN_DIMS:
        raise ValueError("mtrouter_checkpoint_q_hidden_dims_mismatch")
    if q_config.get("dropout") != Q_DROPOUT:
        raise ValueError("mtrouter_checkpoint_q_dropout_mismatch")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("mtrouter_checkpoint_state_dict_missing")
    state_keys = set(state_dict)
    if any(
        key.startswith(("q_network.score_head.", "q_network.cost_head."))
        for key in state_keys
    ):
        raise ValueError("mtrouter_checkpoint_dual_head_rejected")
    if not {
        "q_network.head.weight",
        "q_network.head.bias",
        "model_encoder.residual_embed.weight",
    }.issubset(state_keys):
        raise ValueError("mtrouter_checkpoint_single_head_state_missing")
    _validate_checkpoint_training_provenance(checkpoint)


def _validate_checkpoint_training_provenance(
    checkpoint: Mapping[str, Any],
) -> None:
    receipt = checkpoint.get("papc_training")
    if not isinstance(receipt, Mapping):
        raise ValueError("mtrouter_checkpoint_training_receipt_missing")
    expected_exact = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "upstream_revision": MTROUTER_UPSTREAM_REVISION,
        "upstream_source_sha256s": dict(
            sorted(MTROUTER_TRAINING_SOURCE_SHA256S.items())
        ),
        "training_config": asdict(PUBLISHED_TRAINING_CONFIG),
        "training_config_sha256": PUBLISHED_TRAINING_CONFIG.sha256,
        "training_scope": "router_fit_only",
        "qwen_encoder_model": HISTORY_ENCODER_MODEL,
        "qwen_encoder_frozen": True,
        "history_backend": TRAINING_HISTORY_BACKEND,
        "local_files_only": True,
        "provider_calls": 0,
    }
    for key, expected in expected_exact.items():
        if receipt.get(key) != expected:
            raise ValueError(
                f"mtrouter_checkpoint_training_provenance_mismatch:{key}"
            )
    for key in (
        "fit_registry_sha256",
        "fit_membership_sha256",
        "fit_rows_sha256",
        "fit_internal_split_sha256",
        "qwen_source_sha256",
    ):
        _required_sha256(receipt.get(key), f"checkpoint_{key}")
    surface = receipt.get("surface")
    if not isinstance(surface, str) or not surface.strip():
        raise ValueError("mtrouter_checkpoint_training_surface_invalid")
    backend_name = receipt.get("backend_name")
    if not isinstance(backend_name, str) or not backend_name:
        raise ValueError("mtrouter_checkpoint_training_backend_invalid")
    epoch = receipt.get("epoch")
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or not 1 <= epoch <= TRAINING_EPOCHS
    ):
        raise ValueError("mtrouter_checkpoint_training_epoch_invalid")
    if receipt.get("origin") not in {
        "fit_internal_best",
        "periodic",
        "final",
    }:
        raise ValueError("mtrouter_checkpoint_training_origin_invalid")
    _nonnegative_finite_float(
        receipt.get("train_loss"),
        "mtrouter_checkpoint_training_train_loss",
    )
    fit_val_loss = receipt.get("fit_internal_val_loss")
    if fit_val_loss is not None:
        _nonnegative_finite_float(
            fit_val_loss,
            "mtrouter_checkpoint_training_fit_val_loss",
        )


def _build_torch_modules(torch: Any) -> tuple[Any, Any]:
    nn = torch.nn

    class _ModelEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("meta_mean", torch.zeros(MODEL_META_DIM))
            self.register_buffer("meta_std", torch.zeros(MODEL_META_DIM))
            self.register_buffer(
                "all_metas",
                torch.zeros(len(PUBLISHED_CONFIG.model_names), MODEL_META_DIM),
            )
            self.meta_mlp = nn.Sequential(
                nn.Linear(MODEL_META_DIM, MODEL_META_HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(MODEL_META_HIDDEN_DIM, MODEL_META_EMBED_DIM),
            )
            self.residual_embed = nn.Embedding(
                len(PUBLISHED_CONFIG.model_names),
                MODEL_RESIDUAL_EMBED_DIM,
            )
            self.output_proj = nn.Sequential(
                nn.Linear(
                    MODEL_META_EMBED_DIM + MODEL_RESIDUAL_EMBED_DIM,
                    MODEL_OUTPUT_DIM,
                ),
                nn.LayerNorm(MODEL_OUTPUT_DIM),
            )

        def forward_all(self) -> Any:
            meta = self.meta_mlp(self.all_metas)
            indices = torch.arange(
                len(PUBLISHED_CONFIG.model_names),
                device=self.all_metas.device,
            )
            residual = self.residual_embed(indices)
            return self.output_proj(torch.cat([meta, residual], dim=-1))

        def forward_indices(self, indices: Any) -> Any:
            meta = self.meta_mlp(self.all_metas[indices])
            residual = self.residual_embed(indices)
            return self.output_proj(torch.cat([meta, residual], dim=-1))

    class _QNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(HISTORY_ENCODER_DIM + MODEL_OUTPUT_DIM, 256),
                nn.ReLU(),
                nn.Dropout(Q_DROPOUT),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(Q_DROPOUT),
            )
            self.head = nn.Linear(256, 1)

        def forward(self, state: Any, model: Any) -> Any:
            combined = torch.cat([state, model], dim=-1)
            return self.head(self.backbone(combined))

    return _ModelEncoder(), _QNetwork()


def _validate_corpus(
    trajectories: Sequence[MTRouterTrainingTrajectory],
    *,
    split: str,
) -> tuple[MTRouterTrainingTrajectory, ...]:
    if isinstance(trajectories, (str, bytes)) or not isinstance(
        trajectories,
        Sequence,
    ):
        raise TypeError(f"mtrouter_{split}_corpus_must_be_sequence")
    result = tuple(trajectories)
    if not result:
        raise ValueError(f"mtrouter_{split}_corpus_empty")
    if any(not isinstance(item, MTRouterTrainingTrajectory) for item in result):
        raise TypeError(f"mtrouter_{split}_trajectory_invalid")
    episodes = tuple(item.episode_sha256 for item in result)
    if len(set(episodes)) != len(episodes):
        raise ValueError(f"mtrouter_{split}_episode_duplicate")
    roles = {
        row.exploration.selected_role
        for trajectory in result
        for row in trajectory.rows
    }
    if roles != {"cheap", "strong"}:
        raise ValueError(f"mtrouter_{split}_roulette_role_support_incomplete")
    return tuple(sorted(result, key=lambda item: item.episode_sha256))


def _trajectory_seed(
    surface: str,
    example_sha256: str,
    *,
    split: str,
    trajectory_index: int,
) -> int:
    if split not in {"fit", "dev"}:
        raise ValueError("mtrouter_seed_split_invalid")
    payload = (
        "papc.mtrouter.seed.v1",
        surface,
        example_sha256,
        trajectory_index,
    )
    return int.from_bytes(
        hashlib.sha256(_canonical_json(payload).encode("utf-8")).digest()[:8],
        "big",
    )


def _seed_schedule_sha256(
    manifest: SplitManifest,
    *,
    surface: str,
    split: str,
) -> str:
    return _digest(
        {
            "protocol": ROULETTE_SEED_PROTOCOL,
            "surface": surface,
            "split": split,
            "seeds": {
                example_sha256: [
                    _trajectory_seed(
                        surface,
                        example_sha256,
                        split=split,
                        trajectory_index=index,
                    )
                    for index in range(TRAJECTORIES_PER_TASK)
                ]
                for example_sha256 in manifest.example_sha256s
            },
        }
    )


def _validate_seeded_registry_corpus(
    trajectories: Sequence[MTRouterTrainingTrajectory],
    *,
    manifest: SplitManifest,
    surface: str,
    split: str,
) -> None:
    expected_role = "router_fit" if split == "fit" else "router_dev"
    if manifest.role != expected_role:
        raise ValueError(f"mtrouter_{split}_manifest_role_invalid")
    grouped: dict[str, list[MTRouterTrainingTrajectory]] = {}
    for trajectory in trajectories:
        grouped.setdefault(trajectory.example_sha256, []).append(trajectory)
    _require_exact_membership(
        manifest,
        tuple(grouped),
        field_name=f"mtrouter_{split}",
    )
    for example_sha256, example_trajectories in grouped.items():
        if len(example_trajectories) != TRAJECTORIES_PER_TASK:
            raise ValueError(
                f"mtrouter_{split}_trajectories_per_task_mismatch"
            )
        expected_seeds = {
            _trajectory_seed(
                surface,
                example_sha256,
                split=split,
                trajectory_index=index,
            )
            for index in range(TRAJECTORIES_PER_TASK)
        }
        actual_seeds = {
            trajectory.trajectory_seed
            for trajectory in example_trajectories
        }
        if actual_seeds != expected_seeds:
            raise ValueError(f"mtrouter_{split}_seed_schedule_mismatch")


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
        field_name="mtrouter_membership",
    )
    return _digest(
        {
            "role": manifest.role,
            "registry_sha256": manifest.registry_sha256,
            "example_sha256s": manifest.example_sha256s,
        }
    )


def _normalize_messages(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], ...]:
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise TypeError("mtrouter_messages_must_be_sequence")
    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"mtrouter_message_{index}_must_be_mapping")
        if set(message) != {"role", "content"}:
            raise ValueError(f"mtrouter_message_{index}_schema_invalid")
        role = message["role"]
        content = message["content"]
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"mtrouter_message_{index}_role_invalid")
        if not isinstance(content, str):
            raise TypeError(f"mtrouter_message_{index}_content_must_be_text")
        normalized.append({"role": role, "content": content})
    if not normalized:
        raise ValueError("mtrouter_messages_empty")
    return tuple(normalized)


def _extract_task_description(messages: Sequence[Mapping[str, str]]) -> str:
    for message in messages:
        if message["role"] != "user":
            continue
        content = message["content"]
        match = re.search(
            r"<task_description>(.*?)</task_description>",
            content,
            re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        if "## HLE Question" in content:
            after = content.split("## HLE Question", 1)[1].strip()
            parts = re.split(r"\n\s*---\s*\n", after, maxsplit=1)
            question = (parts[0] if parts else after).strip()
            if question:
                return question
        match = re.search(r"Your task is to[^.]+\.", content)
        if match:
            return match.group(0)
        return content.strip()
    return ""


def _extract_initial_observation(content: str) -> str:
    match = re.search(
        r"<initial_observation>(.*?)</initial_observation>",
        content,
        re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def _count_tokens(text: str, tokenizer: MTRouterTokenizer) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def _truncate_to_tokens(
    text: str,
    max_tokens: int,
    tokenizer: MTRouterTokenizer,
) -> str:
    if not text:
        return ""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens])


def _truncate_recent_context(
    messages: Sequence[Mapping[str, str]],
    *,
    max_tokens: int,
    tokenizer: MTRouterTokenizer,
) -> str:
    if not messages or max_tokens <= 0:
        return ""
    total_turns = sum(message["role"] == "user" for message in messages)
    selected: list[tuple[Mapping[str, str], int]] = []
    total_tokens = 0
    current_turn = total_turns
    for message in reversed(messages):
        tokens = (
            _count_tokens(message["content"], tokenizer)
            + _FORMAT_OVERHEAD_PER_MESSAGE
        )
        if total_tokens + tokens > max_tokens:
            break
        selected.append((message, current_turn))
        total_tokens += tokens
        if message["role"] == "user":
            current_turn -= 1
    return "\n\n".join(
        f"[Turn {turn} - {message['role']}]\n{message['content']}"
        for message, turn in reversed(selected)
    )


def _strip_state_prefix(
    state_dict: Mapping[str, Any],
    prefix: str,
) -> dict[str, Any]:
    return {
        key.removeprefix(prefix): value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def _required_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"mtrouter_{field_name}_invalid")
    return value


def _unit_float(value: Any, field_name: str) -> float:
    resolved = _finite_float(value, field_name)
    if not 0.0 <= resolved <= 1.0:
        raise ValueError(f"{field_name}_outside_unit_interval")
    return resolved


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name}_must_be_numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field_name}_not_finite")
    return resolved


def _nonnegative_finite_float(value: Any, field_name: str) -> float:
    resolved = _finite_float(value, field_name)
    if resolved < 0.0:
        raise ValueError(f"{field_name}_negative")
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

"""Fail-closed adapter and fit-only trainer for pinned OpenSquilla.

The adapter executes OpenSquilla's local ``V4Phase3Strategy`` exactly from
revision ``7e3965e...``.  The Table 1 path continues the shipped four-class
LightGBM head from paired in-domain fit logs using the upstream self-learning
trainer; the shipped cold-start head is never accepted as a main-table
artifact.  This module has no provider-facing code.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import shutil
import struct
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Literal, Protocol

from papc.baselines.router_contract import (
    CHEAP_MODEL,
    OPENSQUILLA_G0,
    STRONG_MODEL,
    ModelRole,
    RouterArtifactBinding,
    SplitManifest,
    assert_disjoint_manifests,
)

OPENSQUILLA_UPSTREAM_REVISION = (
    "7e3965e7cc64316be87dc0d535eace39d22a2c93"
)
OPENSQUILLA_STRATEGY = (
    "opensquilla.squilla_router.v4_phase3.V4Phase3Strategy"
)
OPENSQUILLA_BUNDLE_RELATIVE_PATH = Path(
    "src/opensquilla/squilla_router/models/v4.2_phase3_inference"
)
OPENSQUILLA_STRATEGY_RELATIVE_PATH = Path(
    "src/opensquilla/squilla_router/v4_phase3.py"
)
OPENSQUILLA_VALID_TIERS = ("c0", "c1", "c2", "c3")
OPENSQUILLA_ROUTE_CLASSES = ("R0", "R1", "R2", "R3")
OPENSQUILLA_FEATURE_DIM = 390
OPENSQUILLA_FIT_BUNDLE_RELATIVE_PATH = Path("bundle")
OPENSQUILLA_FIT_LABEL_RULE = (
    "paired_official_fit_outcomes:"
    "cheap_success_to_R0;"
    "cheap_failure_strong_success_to_R2;"
    "both_failure_to_R3"
)
OPENSQUILLA_FIT_WEIGHT_RULE = (
    "each_fit_task_has_unit_total_weight_split_equally_across_its_turns"
)
OPENSQUILLA_TRAINING_NUMPY_VERSION = "1.26.4"
OPENSQUILLA_FIT_CONSUMPTION = (
    "in_domain_fit_only_paired_official_outcomes"
)
OPENSQUILLA_TRAINING_EXPORT_SCHEMA = (
    "papc.opensquilla.fit_training_export.v1"
)
OPENSQUILLA_RUNTIME_ARTIFACT_SCHEMA = (
    "papc.opensquilla.runtime_artifact.v2"
)

# These are the exact upstream source files exercised by feature capture,
# incremental head training, and bundle assembly.  The copied runtime feature
# sources also remain covered by the full base-bundle tree digest.
OPENSQUILLA_TRAINING_SOURCE_SHA256S = {
    "src/opensquilla/squilla_router/v4_phase3.py": (
        "44226245df38f32acda48729d97e958cc148182eaccf92e06c91671ce1afce46"
    ),
    "src/opensquilla/squilla_router/self_learning/train.py": (
        "176dfe85a601880bb6c05aa66909ee703d4fac4e32917a70d53bddeea401832d"
    ),
    "src/opensquilla/squilla_router/self_learning/schema.py": (
        "ddce89762fde84442bc45aaff6d0c4b37df2d5f95f2a94c39ad6ccaa7420b251"
    ),
    (
        "src/opensquilla/squilla_router/models/"
        "v4.2_phase3_inference/runtime_src/src/router/inference/core.py"
    ): (
        "253f354f6ddbc482c465df3a6fca4bcfba3ab2765e49f0991a219d87fb49f072"
    ),
    (
        "src/opensquilla/squilla_router/models/"
        "v4.2_phase3_inference/runtime_src/src/router/inference/features.py"
    ): (
        "c51cc7f351394921d9821a2a03fbc6d9c78f0913d5172dccca27c73b59da246d"
    ),
}
OPENSQUILLA_TRAINER_SOURCE_SHA256 = hashlib.sha256(
    json.dumps(
        OPENSQUILLA_TRAINING_SOURCE_SHA256S,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

_SERIALIZER_CONFIG = {
    "schema": "papc.opensquilla.public_turn.v1",
    "current_user_text": "required_utf8",
    "recent_user_turns": {
        "order": "oldest_to_newest",
        "maximum": 4,
    },
    "previous_assistant_text": "optional_utf8",
    "previous_assistant_usage": (
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cached_tokens",
        "duration_ms",
    ),
    "prior_route_decisions": {
        "maximum": 5,
        "fields": ("route_class", "difficulty", "margin"),
    },
    "public_context_metadata": (
        "turn_index",
        "context_tokens_est",
        "n_tools",
        "tool_result_length",
        "has_code_block",
        "has_file_reference",
        "has_url",
        "has_tool_results",
    ),
    "flags_text": "current_user_text",
}
OPENSQUILLA_SERIALIZER_CONFIG_SHA256 = hashlib.sha256(
    json.dumps(
        _SERIALIZER_CONFIG,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

_RUNTIME_DEPENDENCIES = (
    "joblib",
    "lightgbm",
    "numpy",
    "onnxruntime",
    "sklearn",
    "structlog",
    "tokenizers",
    "yaml",
)
_REQUIRED_BUNDLE_PATHS = (
    "artifact_manifest.json",
    "inference_manifest.json",
    "router.runtime.yaml",
    "version.json",
    "lgbm_main.bin",
    "lgbm_aux.bin",
    "mlp/model.onnx",
    "mlp/scaler.joblib",
    "features/bge_pca.joblib",
    "features/config.pkl",
    "features/meta.json",
    "features/svd.pkl",
    "features/tfidf.pkl",
    "bge_onnx/model.onnx",
    "bge_onnx/tokenizer.json",
    "runtime_src/src/router/inference/core.py",
    "runtime_src/src/router/inference/features.py",
    "runtime_src/src/router/inference/postprocess.py",
    "runtime_src/src/router/inference/types.py",
)
_USAGE_FIELDS = frozenset(
    {
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cached_tokens",
        "duration_ms",
    }
)

RouteClass = Literal["R0", "R1", "R2", "R3"]


@dataclass(frozen=True)
class OpenSquillaPriorRouteDecision:
    route_class: RouteClass
    difficulty: float
    margin: float

    def __post_init__(self) -> None:
        if self.route_class not in OPENSQUILLA_ROUTE_CLASSES:
            raise ValueError("opensquilla_prior_route_class_invalid")
        object.__setattr__(
            self,
            "difficulty",
            _finite_float(self.difficulty, "opensquilla_prior_difficulty"),
        )
        object.__setattr__(
            self,
            "margin",
            _finite_float(self.margin, "opensquilla_prior_margin"),
        )


@dataclass(frozen=True)
class OpenSquillaPublicContext:
    turn_index: int
    context_tokens_est: int
    n_tools: int
    tool_result_length: int
    has_code_block: bool
    has_file_reference: bool
    has_url: bool
    has_tool_results: bool

    def __post_init__(self) -> None:
        for name in (
            "turn_index",
            "context_tokens_est",
            "n_tools",
            "tool_result_length",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"opensquilla_context_{name}_invalid")
        for name in (
            "has_code_block",
            "has_file_reference",
            "has_url",
            "has_tool_results",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"opensquilla_context_{name}_must_be_bool")

    def public_record(self) -> dict[str, int | bool]:
        return asdict(self)


@dataclass(frozen=True)
class OpenSquillaTurnInput:
    """Every public channel consumed by the pinned 390-d feature builder."""

    current_user_text: str = field(repr=False)
    recent_user_turns: tuple[str, ...] = field(repr=False)
    previous_assistant_text: str | None = field(repr=False)
    previous_assistant_usage: Mapping[str, int | float] = field(repr=False)
    prior_route_decisions: tuple[OpenSquillaPriorRouteDecision, ...]
    public_context_metadata: OpenSquillaPublicContext
    _canonical_record: dict[str, Any] = field(init=False, repr=False)
    _sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        current = _required_text(
            self.current_user_text,
            "opensquilla_current_user_text",
        )
        recent = tuple(
            _required_text(text, "opensquilla_recent_user_turn")
            for text in self.recent_user_turns
        )
        if len(recent) > 4:
            raise ValueError("opensquilla_recent_user_turns_exceed_upstream_window")
        previous = self.previous_assistant_text
        if previous is not None:
            previous = _required_text(
                previous,
                "opensquilla_previous_assistant_text",
            )
        usage = _validated_usage(self.previous_assistant_usage)
        if previous is None and usage:
            raise ValueError("opensquilla_usage_without_previous_assistant")
        decisions = tuple(self.prior_route_decisions)
        if len(decisions) > 5:
            raise ValueError("opensquilla_prior_decisions_exceed_upstream_window")
        if not all(
            isinstance(item, OpenSquillaPriorRouteDecision)
            for item in decisions
        ):
            raise TypeError("opensquilla_prior_route_decision_invalid")
        if not isinstance(
            self.public_context_metadata,
            OpenSquillaPublicContext,
        ):
            raise TypeError("opensquilla_public_context_invalid")

        record = {
            "current_user_text": current,
            "recent_user_turns": recent,
            "previous_assistant_text": previous,
            "previous_assistant_usage": usage,
            "prior_route_decisions": [
                asdict(decision) for decision in decisions
            ],
            "public_context_metadata": (
                self.public_context_metadata.public_record()
            ),
        }
        object.__setattr__(self, "current_user_text", current)
        object.__setattr__(self, "recent_user_turns", recent)
        object.__setattr__(self, "previous_assistant_text", previous)
        object.__setattr__(self, "previous_assistant_usage", usage)
        object.__setattr__(self, "prior_route_decisions", decisions)
        object.__setattr__(self, "_canonical_record", record)
        object.__setattr__(
            self,
            "_sha256",
            hashlib.sha256(_canonical_bytes(record)).hexdigest(),
        )

    @property
    def sha256(self) -> str:
        return self._sha256

    def public_record(self) -> dict[str, Any]:
        return json.loads(_canonical_bytes(self._canonical_record))


@dataclass(frozen=True)
class OpenSquillaPairedFitEpisode:
    """One fit-only episode with public turns and paired official outcomes."""

    example_sha256: str
    cheap_turns: tuple[OpenSquillaTurnInput, ...]
    strong_turns: tuple[OpenSquillaTurnInput, ...]
    cheap_official_success: bool
    strong_official_success: bool
    cheap_episode_receipt_sha256: str
    strong_episode_receipt_sha256: str

    def __post_init__(self) -> None:
        _required_sha256(
            self.example_sha256,
            "opensquilla_fit_example_sha256",
        )
        for role in ("cheap", "strong"):
            field_name = f"{role}_turns"
            turns = tuple(getattr(self, field_name))
            if not turns or any(
                not isinstance(turn, OpenSquillaTurnInput) for turn in turns
            ):
                raise ValueError(
                    f"opensquilla_fit_{role}_turns_invalid"
                )
            observed_indices = tuple(
                turn.public_context_metadata.turn_index for turn in turns
            )
            if observed_indices != tuple(range(len(turns))):
                raise ValueError(
                    f"opensquilla_fit_{role}_turn_indices_not_contiguous"
                )
            if len({turn.sha256 for turn in turns}) != len(turns):
                raise ValueError(
                    f"opensquilla_fit_{role}_turn_duplicate"
                )
            object.__setattr__(self, field_name, turns)
        for field_name in (
            "cheap_official_success",
            "strong_official_success",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"opensquilla_fit_{field_name}_must_be_bool")
        for field_name in (
            "cheap_episode_receipt_sha256",
            "strong_episode_receipt_sha256",
        ):
            _required_sha256(getattr(self, field_name), field_name)
        if (
            self.cheap_episode_receipt_sha256
            == self.strong_episode_receipt_sha256
        ):
            raise ValueError("opensquilla_fit_paired_receipts_reused")

    @property
    def target_route_index(self) -> int:
        # R0 is the lowest capable cheap class.  R2 is the lowest class on the
        # strong side of the four-tier router, while R3 preserves the upstream
        # strongest-class signal when neither endpoint succeeds.
        if self.cheap_official_success:
            return 0
        if self.strong_official_success:
            return 2
        return 3

    def source_record(self) -> dict[str, Any]:
        return {
            "example_sha256": self.example_sha256,
            "cheap_turn_input_sha256s": [
                turn.sha256 for turn in self.cheap_turns
            ],
            "strong_turn_input_sha256s": [
                turn.sha256 for turn in self.strong_turns
            ],
            "cheap_official_success": self.cheap_official_success,
            "strong_official_success": self.strong_official_success,
            "cheap_episode_receipt_sha256": (
                self.cheap_episode_receipt_sha256
            ),
            "strong_episode_receipt_sha256": (
                self.strong_episode_receipt_sha256
            ),
            "target_route_class": (
                OPENSQUILLA_ROUTE_CLASSES[self.target_route_index]
            ),
        }


class OpenSquillaFeatureCapture(Protocol):
    """Exact upstream 390-d feature capture surface used by fit adaptation."""

    bundle_sha256: str
    feature_schema_version: str

    def capture(
        self,
        turn: OpenSquillaTurnInput,
    ) -> tuple[float, ...]: ...


@dataclass(frozen=True)
class OpenSquillaCapturedFitRow:
    example_sha256: str
    source_role: ModelRole
    source_episode_receipt_sha256: str
    turn_index: int
    public_input_sha256: str
    feature_sha256: str
    target_route_index: int
    sample_weight: float
    features_390: tuple[float, ...] = field(repr=False)

    def __post_init__(self) -> None:
        for field_name in (
            "example_sha256",
            "source_episode_receipt_sha256",
            "public_input_sha256",
            "feature_sha256",
        ):
            _required_sha256(getattr(self, field_name), field_name)
        if self.source_role not in {"cheap", "strong"}:
            raise ValueError("opensquilla_fit_row_source_role_invalid")
        if (
            isinstance(self.turn_index, bool)
            or not isinstance(self.turn_index, int)
            or self.turn_index < 0
        ):
            raise ValueError("opensquilla_fit_row_turn_index_invalid")
        if (
            isinstance(self.target_route_index, bool)
            or self.target_route_index not in {0, 1, 2, 3}
        ):
            raise ValueError("opensquilla_fit_row_target_invalid")
        weight = _finite_float(
            self.sample_weight,
            "opensquilla_fit_row_weight",
        )
        if weight <= 0.0:
            raise ValueError("opensquilla_fit_row_weight_not_positive")
        object.__setattr__(self, "sample_weight", weight)
        features = _validated_feature_vector(self.features_390)
        if _feature_sha256(features) != self.feature_sha256:
            raise ValueError("opensquilla_fit_row_feature_digest_mismatch")
        object.__setattr__(self, "features_390", features)

    def public_record(self) -> dict[str, Any]:
        return {
            "example_sha256": self.example_sha256,
            "source_role": self.source_role,
            "source_episode_receipt_sha256": (
                self.source_episode_receipt_sha256
            ),
            "turn_index": self.turn_index,
            "public_input_sha256": self.public_input_sha256,
            "feature_sha256": self.feature_sha256,
            "target_route_index": self.target_route_index,
            "target_route_class": (
                OPENSQUILLA_ROUTE_CLASSES[self.target_route_index]
            ),
            "sample_weight": self.sample_weight,
        }


@dataclass(frozen=True)
class OpenSquillaFitCorpus:
    """Fit-only 390-d matrix plus exact membership and outcome provenance."""

    fit_registry_sha256: str
    fit_membership_sha256: str
    base_bundle_sha256: str
    feature_schema_version: str
    source_episode_sha256: str
    fit_example_sha256s: tuple[str, ...]
    rows: tuple[OpenSquillaCapturedFitRow, ...]
    paired_fit_log_count: int
    label_rule: str = OPENSQUILLA_FIT_LABEL_RULE
    weight_rule: str = OPENSQUILLA_FIT_WEIGHT_RULE

    def __post_init__(self) -> None:
        for field_name in (
            "fit_registry_sha256",
            "fit_membership_sha256",
            "base_bundle_sha256",
            "source_episode_sha256",
        ):
            _required_sha256(getattr(self, field_name), field_name)
        if (
            not isinstance(self.feature_schema_version, str)
            or not self.feature_schema_version
            or self.feature_schema_version == "unknown"
        ):
            raise ValueError("opensquilla_feature_schema_version_invalid")
        examples = tuple(self.fit_example_sha256s)
        if not examples or len(examples) != len(set(examples)):
            raise ValueError("opensquilla_fit_corpus_examples_invalid")
        for value in examples:
            _required_sha256(value, "opensquilla_fit_corpus_example")
        object.__setattr__(self, "fit_example_sha256s", examples)
        rows = tuple(self.rows)
        if not rows or any(
            not isinstance(row, OpenSquillaCapturedFitRow) for row in rows
        ):
            raise ValueError("opensquilla_fit_corpus_rows_invalid")
        object.__setattr__(self, "rows", rows)
        if self.paired_fit_log_count != len(examples):
            raise ValueError("opensquilla_fit_log_count_mismatch")
        if {row.example_sha256 for row in rows} != set(examples):
            raise ValueError("opensquilla_fit_corpus_membership_mismatch")
        for example_sha256 in examples:
            task_weight = sum(
                row.sample_weight
                for row in rows
                if row.example_sha256 == example_sha256
            )
            if not math.isclose(
                task_weight,
                1.0,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "opensquilla_fit_task_weight_not_normalized"
                )
        if self.label_rule != OPENSQUILLA_FIT_LABEL_RULE:
            raise ValueError("opensquilla_fit_label_rule_mismatch")
        if self.weight_rule != OPENSQUILLA_FIT_WEIGHT_RULE:
            raise ValueError("opensquilla_fit_weight_rule_mismatch")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            _canonical_bytes(self.public_record())
        ).hexdigest()

    @property
    def class_distribution(self) -> dict[int, int]:
        return dict(
            sorted(Counter(row.target_route_index for row in self.rows).items())
        )

    def public_record(self) -> dict[str, Any]:
        return {
            "fit_registry_sha256": self.fit_registry_sha256,
            "fit_membership_sha256": self.fit_membership_sha256,
            "base_bundle_sha256": self.base_bundle_sha256,
            "feature_schema_version": self.feature_schema_version,
            "source_episode_sha256": self.source_episode_sha256,
            "fit_example_sha256s": list(self.fit_example_sha256s),
            "rows": [row.public_record() for row in self.rows],
            "paired_fit_log_count": self.paired_fit_log_count,
            "fit_turn_count": len(self.rows),
            "class_distribution": self.class_distribution,
            "label_rule": self.label_rule,
            "weight_rule": self.weight_rule,
        }


@dataclass(frozen=True)
class OpenSquillaPublishedTrainingConfig:
    """Pinned upstream self-learning defaults with their deterministic seed."""

    objective: str = "multiclass"
    num_class: int = 4
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_data_in_leaf: int = 5
    num_boost_round: int = 60
    deterministic: bool = True
    force_col_wise: bool = True
    lightgbm_seed: int = 0
    lightgbm_seed_source: str = "pinned_upstream_lightgbm_default"
    numpy_version: str = OPENSQUILLA_TRAINING_NUMPY_VERSION
    base_initialization: str = "shipped_lgbm_main.bin_via_init_model"
    label_rule: str = OPENSQUILLA_FIT_LABEL_RULE
    weight_rule: str = OPENSQUILLA_FIT_WEIGHT_RULE
    upstream_revision: str = OPENSQUILLA_UPSTREAM_REVISION

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            _canonical_bytes(asdict(self))
        ).hexdigest()


OPENSQUILLA_TRAINING_CONFIG = OpenSquillaPublishedTrainingConfig()


@dataclass(frozen=True)
class OpenSquillaTrainingExport:
    """Complete fit-adapted bundle and source-bound training receipt."""

    surface: str
    fit_registry_sha256: str
    fit_membership_sha256: str
    fit_corpus_sha256: str
    source_episode_sha256: str
    fit_example_sha256s: tuple[str, ...]
    in_domain_fit_log_count: int
    fit_turn_count: int
    feature_schema_version: str
    base_bundle_sha256: str
    adapted_bundle_sha256: str
    base_lgbm_sha256: str
    adapted_lgbm_sha256: str
    training_config_sha256: str
    trainer_source_sha256: str
    upstream_source_sha256s: tuple[tuple[str, str], ...]
    numpy_version: str
    scikit_learn_version: str
    lightgbm_version: str
    class_distribution: tuple[tuple[int, int], ...]
    used_init_model: bool
    provider_calls: int = 0
    bundle_relative_path: str = str(
        OPENSQUILLA_FIT_BUNDLE_RELATIVE_PATH
    )
    schema: str = OPENSQUILLA_TRAINING_EXPORT_SCHEMA
    upstream_revision: str = OPENSQUILLA_UPSTREAM_REVISION
    fit_consumption: str = OPENSQUILLA_FIT_CONSUMPTION

    def __post_init__(self) -> None:
        if not isinstance(self.surface, str) or not self.surface.strip():
            raise ValueError("opensquilla_training_surface_invalid")
        for field_name in (
            "fit_registry_sha256",
            "fit_membership_sha256",
            "fit_corpus_sha256",
            "source_episode_sha256",
            "base_bundle_sha256",
            "adapted_bundle_sha256",
            "base_lgbm_sha256",
            "adapted_lgbm_sha256",
            "training_config_sha256",
            "trainer_source_sha256",
        ):
            _required_sha256(getattr(self, field_name), field_name)
        examples = tuple(self.fit_example_sha256s)
        if not examples or len(examples) != len(set(examples)):
            raise ValueError("opensquilla_training_fit_examples_invalid")
        for value in examples:
            _required_sha256(value, "opensquilla_training_fit_example")
        object.__setattr__(self, "fit_example_sha256s", examples)
        if self.in_domain_fit_log_count != len(examples):
            raise ValueError("opensquilla_training_fit_log_count_mismatch")
        if (
            isinstance(self.fit_turn_count, bool)
            or not isinstance(self.fit_turn_count, int)
            or self.fit_turn_count < self.in_domain_fit_log_count
        ):
            raise ValueError("opensquilla_training_fit_turn_count_invalid")
        if (
            not isinstance(self.feature_schema_version, str)
            or not self.feature_schema_version
            or self.feature_schema_version == "unknown"
        ):
            raise ValueError("opensquilla_training_feature_schema_invalid")
        if self.training_config_sha256 != (
            OPENSQUILLA_TRAINING_CONFIG.sha256
        ):
            raise ValueError("opensquilla_training_config_mismatch")
        if self.trainer_source_sha256 != (
            OPENSQUILLA_TRAINER_SOURCE_SHA256
        ):
            raise ValueError("opensquilla_training_source_mismatch")
        sources = tuple(tuple(item) for item in self.upstream_source_sha256s)
        if dict(sources) != OPENSQUILLA_TRAINING_SOURCE_SHA256S:
            raise ValueError("opensquilla_training_sources_mismatch")
        object.__setattr__(self, "upstream_source_sha256s", sources)
        distribution = tuple(tuple(item) for item in self.class_distribution)
        if (
            not distribution
            or any(
                len(item) != 2
                or item[0] not in {0, 1, 2, 3}
                or isinstance(item[1], bool)
                or not isinstance(item[1], int)
                or item[1] <= 0
                for item in distribution
            )
            or sum(count for _, count in distribution)
            != self.fit_turn_count
        ):
            raise ValueError(
                "opensquilla_training_class_distribution_invalid"
            )
        object.__setattr__(self, "class_distribution", distribution)
        if self.used_init_model is not True:
            raise ValueError("opensquilla_training_requires_init_model")
        if self.provider_calls != 0:
            raise ValueError("opensquilla_training_provider_calls_nonzero")
        if self.bundle_relative_path != str(
            OPENSQUILLA_FIT_BUNDLE_RELATIVE_PATH
        ):
            raise ValueError("opensquilla_training_bundle_path_invalid")
        if self.schema != OPENSQUILLA_TRAINING_EXPORT_SCHEMA:
            raise ValueError("opensquilla_training_schema_invalid")
        if self.upstream_revision != OPENSQUILLA_UPSTREAM_REVISION:
            raise ValueError("opensquilla_training_revision_mismatch")
        if self.fit_consumption != OPENSQUILLA_FIT_CONSUMPTION:
            raise ValueError("opensquilla_training_fit_consumption_invalid")
        if (
            not isinstance(self.lightgbm_version, str)
            or not self.lightgbm_version
        ):
            raise ValueError("opensquilla_training_lightgbm_version_invalid")
        if self.numpy_version != OPENSQUILLA_TRAINING_NUMPY_VERSION:
            raise ValueError("opensquilla_training_numpy_version_mismatch")
        if (
            not isinstance(self.scikit_learn_version, str)
            or not self.scikit_learn_version
        ):
            raise ValueError("opensquilla_training_sklearn_version_invalid")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            _canonical_bytes(self.public_record())
        ).hexdigest()

    def public_record(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "fit_example_sha256s": list(self.fit_example_sha256s),
            "upstream_source_sha256s": [
                list(item) for item in self.upstream_source_sha256s
            ],
            "class_distribution": [
                list(item) for item in self.class_distribution
            ],
        }

    def json_record(self) -> dict[str, Any]:
        return {
            **self.public_record(),
            "training_export_sha256": self.sha256,
        }


@dataclass(frozen=True)
class OpenSquillaDevProjection:
    """Dev-frozen monotone map from OpenSquilla's four tiers to two roles."""

    strong_class_floor: Literal[0, 1, 2, 3, 4]
    dev_registry_sha256: str
    dev_membership_sha256: str
    selection_report_sha256: str
    strong_full_dev_anchor: float
    method: str = OPENSQUILLA_G0
    serializer_config_sha256: str = OPENSQUILLA_SERIALIZER_CONFIG_SHA256

    def __post_init__(self) -> None:
        if (
            isinstance(self.strong_class_floor, bool)
            or self.strong_class_floor not in {0, 1, 2, 3, 4}
        ):
            raise ValueError("opensquilla_projection_floor_invalid")
        _required_sha256(
            self.dev_registry_sha256,
            "opensquilla_projection_dev_registry_sha256",
        )
        _required_sha256(
            self.dev_membership_sha256,
            "opensquilla_projection_dev_membership_sha256",
        )
        _required_sha256(
            self.selection_report_sha256,
            "opensquilla_projection_selection_report_sha256",
        )
        if self.method != OPENSQUILLA_G0:
            raise ValueError("opensquilla_projection_method_invalid")
        if (
            self.serializer_config_sha256
            != OPENSQUILLA_SERIALIZER_CONFIG_SHA256
        ):
            raise ValueError("opensquilla_projection_serializer_mismatch")
        object.__setattr__(
            self,
            "strong_full_dev_anchor",
            _unit_float(
                self.strong_full_dev_anchor,
                "opensquilla_projection_strong_full_dev_anchor",
            ),
        )

    @classmethod
    def freeze_from_dev(
        cls,
        *,
        dev_manifest: SplitManifest,
        selection_report: OpenSquillaDevSelectionReport,
    ) -> OpenSquillaDevProjection:
        if not isinstance(dev_manifest, SplitManifest):
            raise TypeError("opensquilla_projection_manifest_invalid")
        if dev_manifest.role != "router_dev":
            raise ValueError("opensquilla_projection_requires_router_dev")
        if not isinstance(selection_report, OpenSquillaDevSelectionReport):
            raise TypeError("opensquilla_selection_report_invalid")
        if (
            selection_report.dev_registry_sha256
            != dev_manifest.registry_sha256
            or selection_report.dev_membership_sha256
            != _membership_sha256(
                dev_manifest,
                dev_manifest.example_sha256s,
            )
        ):
            raise ValueError("opensquilla_selection_report_registry_mismatch")
        return cls(
            strong_class_floor=selection_report.selected_strong_class_floor,
            dev_registry_sha256=dev_manifest.registry_sha256,
            dev_membership_sha256=selection_report.dev_membership_sha256,
            selection_report_sha256=selection_report.sha256,
            strong_full_dev_anchor=(
                selection_report.strong_full_dev_anchor
            ),
        )

    def selected_role(self, route_class: RouteClass) -> ModelRole:
        try:
            route_index = OPENSQUILLA_ROUTE_CLASSES.index(route_class)
        except ValueError as exc:
            raise ValueError("opensquilla_projection_route_class_invalid") from exc
        return (
            "strong"
            if route_index >= self.strong_class_floor
            else "cheap"
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            _canonical_bytes(self.public_record())
        ).hexdigest()

    def public_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OpenSquillaDevEvaluation:
    example_sha256: str
    strong_class_floor: Literal[0, 1, 2, 3, 4]
    official_quality: float
    provider_api_cost_usd: float
    strong_call_count: int

    def __post_init__(self) -> None:
        _required_sha256(self.example_sha256, "opensquilla_dev_example")
        if (
            isinstance(self.strong_class_floor, bool)
            or self.strong_class_floor not in {0, 1, 2, 3, 4}
        ):
            raise ValueError("opensquilla_dev_floor_invalid")
        object.__setattr__(
            self,
            "official_quality",
            _unit_float(
                self.official_quality,
                "opensquilla_dev_official_quality",
            ),
        )
        cost = _finite_float(
            self.provider_api_cost_usd,
            "opensquilla_dev_provider_api_cost",
        )
        if cost < 0.0:
            raise ValueError("opensquilla_dev_provider_api_cost_negative")
        object.__setattr__(self, "provider_api_cost_usd", cost)
        if (
            isinstance(self.strong_call_count, bool)
            or not isinstance(self.strong_call_count, int)
            or self.strong_call_count < 0
        ):
            raise ValueError("opensquilla_dev_strong_call_count_invalid")

    def public_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OpenSquillaDevCandidateReceipt:
    strong_class_floor: Literal[0, 1, 2, 3, 4]
    config_sha256: str
    dev_quality: float
    provider_api_cost_usd: float
    strong_call_count: int
    example_membership_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.strong_class_floor, bool)
            or self.strong_class_floor not in {0, 1, 2, 3, 4}
        ):
            raise ValueError("opensquilla_dev_floor_invalid")
        if self.config_sha256 != _projection_config_sha256(
            self.strong_class_floor
        ):
            raise ValueError("opensquilla_dev_config_sha256_mismatch")
        _required_sha256(
            self.example_membership_sha256,
            "opensquilla_dev_membership",
        )
        object.__setattr__(
            self,
            "dev_quality",
            _unit_float(self.dev_quality, "opensquilla_dev_quality"),
        )
        cost = _finite_float(
            self.provider_api_cost_usd,
            "opensquilla_dev_provider_api_cost",
        )
        if cost < 0.0:
            raise ValueError("opensquilla_dev_provider_api_cost_negative")
        object.__setattr__(self, "provider_api_cost_usd", cost)
        if (
            isinstance(self.strong_call_count, bool)
            or not isinstance(self.strong_call_count, int)
            or self.strong_call_count < 0
        ):
            raise ValueError("opensquilla_dev_strong_call_count_invalid")

    def public_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OpenSquillaDevSelectionReport:
    dev_registry_sha256: str
    dev_membership_sha256: str
    replay_rows_sha256: str
    strong_full_dev_anchor: float
    selected_strong_class_floor: Literal[0, 1, 2, 3, 4]
    candidate_receipts: tuple[OpenSquillaDevCandidateReceipt, ...]
    selection_rule: str = (
        "strong_full_dev_anchor_then_min_provider_cost_then_min_strong_calls_then_config_sha256"
    )

    def __post_init__(self) -> None:
        for field_name in (
            "dev_registry_sha256",
            "dev_membership_sha256",
            "replay_rows_sha256",
        ):
            _required_sha256(getattr(self, field_name), field_name)
        object.__setattr__(
            self,
            "strong_full_dev_anchor",
            _unit_float(
                self.strong_full_dev_anchor,
                "opensquilla_strong_full_dev_anchor",
            ),
        )
        receipts = tuple(self.candidate_receipts)
        if tuple(item.strong_class_floor for item in receipts) != (
            0,
            1,
            2,
            3,
            4,
        ):
            raise ValueError("opensquilla_dev_candidate_family_incomplete")
        if self.selected_strong_class_floor not in {0, 1, 2, 3, 4}:
            raise ValueError("opensquilla_selected_floor_invalid")
        object.__setattr__(self, "candidate_receipts", receipts)
        if self.selection_rule != (
            "strong_full_dev_anchor_then_min_provider_cost_then_min_strong_calls_then_config_sha256"
        ):
            raise ValueError("opensquilla_dev_selection_rule_invalid")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            _canonical_bytes(self.public_record())
        ).hexdigest()

    def public_record(self) -> dict[str, Any]:
        return {
            "dev_registry_sha256": self.dev_registry_sha256,
            "dev_membership_sha256": self.dev_membership_sha256,
            "replay_rows_sha256": self.replay_rows_sha256,
            "strong_full_dev_anchor": self.strong_full_dev_anchor,
            "selected_strong_class_floor": (
                self.selected_strong_class_floor
            ),
            "candidate_receipts": [
                item.public_record() for item in self.candidate_receipts
            ],
            "selection_rule": self.selection_rule,
        }


def select_opensquilla_dev_projection(
    evaluations: tuple[OpenSquillaDevEvaluation, ...],
    *,
    dev_manifest: SplitManifest,
) -> OpenSquillaDevSelectionReport:
    if not isinstance(evaluations, tuple) or not evaluations or any(
        not isinstance(item, OpenSquillaDevEvaluation)
        for item in evaluations
    ):
        raise TypeError("opensquilla_dev_evaluations_must_be_tuple")
    if not isinstance(dev_manifest, SplitManifest):
        raise TypeError("opensquilla_dev_manifest_invalid")
    if dev_manifest.role != "router_dev":
        raise ValueError("opensquilla_dev_manifest_role_invalid")
    receipts: list[OpenSquillaDevCandidateReceipt] = []
    for floor in range(5):
        rows = tuple(
            item for item in evaluations if item.strong_class_floor == floor
        )
        identities = tuple(item.example_sha256 for item in rows)
        _require_exact_membership(
            dev_manifest,
            identities,
            field_name=f"opensquilla_dev_floor_{floor}",
        )
        receipts.append(
            OpenSquillaDevCandidateReceipt(
                strong_class_floor=floor,  # type: ignore[arg-type]
                config_sha256=_projection_config_sha256(floor),
                dev_quality=sum(item.official_quality for item in rows)
                / len(rows),
                provider_api_cost_usd=sum(
                    item.provider_api_cost_usd for item in rows
                ),
                strong_call_count=sum(
                    item.strong_call_count for item in rows
                ),
                example_membership_sha256=_membership_sha256(
                    dev_manifest,
                    identities,
                ),
            )
        )
    anchor = receipts[0].dev_quality
    feasible = tuple(
        item for item in receipts if item.dev_quality + 1e-12 >= anchor
    )
    if not feasible:
        raise ValueError("opensquilla_no_dev_candidate_meets_strong_anchor")
    selected = min(
        feasible,
        key=lambda item: (
            item.provider_api_cost_usd,
            item.strong_call_count,
            item.config_sha256,
        ),
    )
    return OpenSquillaDevSelectionReport(
        dev_registry_sha256=dev_manifest.registry_sha256,
        dev_membership_sha256=_membership_sha256(
            dev_manifest,
            dev_manifest.example_sha256s,
        ),
        replay_rows_sha256=hashlib.sha256(
            _canonical_bytes(
                [item.public_record() for item in evaluations]
            )
        ).hexdigest(),
        strong_full_dev_anchor=anchor,
        selected_strong_class_floor=selected.strong_class_floor,
        candidate_receipts=tuple(receipts),
    )


@dataclass(frozen=True)
class OpenSquillaReadyBlocker:
    code: str
    detail: str


@dataclass(frozen=True)
class OpenSquillaReadiness:
    checkout_root: str
    revision: str | None
    bundle_sha256: str | None
    serializer_config_sha256: str
    blockers: tuple[OpenSquillaReadyBlocker, ...]

    @property
    def ready(self) -> bool:
        return not self.blockers

    @property
    def blocker_codes(self) -> tuple[str, ...]:
        return tuple(blocker.code for blocker in self.blockers)


class OpenSquillaNotReadyError(RuntimeError):
    def __init__(self, readiness: OpenSquillaReadiness) -> None:
        self.readiness = readiness
        detail = "; ".join(
            f"{blocker.code}: {blocker.detail}"
            for blocker in readiness.blockers
        )
        super().__init__(f"opensquilla_upstream_not_ready: {detail}")


@dataclass(frozen=True)
class OpenSquillaPrediction:
    route_class: RouteClass
    selected_role: ModelRole
    tier: str
    confidence: float
    probabilities: tuple[tuple[str, float], ...]
    difficulty: float
    margin: float
    source: str
    public_input_sha256: str
    bundle_sha256: str
    serializer_config_sha256: str
    projection_sha256: str


class OpenSquillaUpstreamFeatureCapture:
    """Fit-only wrapper around the pinned runtime's emitted 390-d vectors."""

    def __init__(
        self,
        *,
        strategy: Any,
        bundle_sha256: str,
        feature_schema_version: str,
    ) -> None:
        self._strategy = strategy
        self.bundle_sha256 = _required_sha256(
            bundle_sha256,
            "opensquilla_capture_bundle_sha256",
        )
        if (
            not isinstance(feature_schema_version, str)
            or not feature_schema_version
            or feature_schema_version == "unknown"
        ):
            raise ValueError("opensquilla_capture_feature_schema_invalid")
        self.feature_schema_version = feature_schema_version

    @classmethod
    def load(
        cls,
        *,
        checkout_root: str | Path,
    ) -> OpenSquillaUpstreamFeatureCapture:
        readiness = inspect_opensquilla_readiness(checkout_root)
        if not readiness.ready or readiness.bundle_sha256 is None:
            raise OpenSquillaNotReadyError(readiness)
        strategy_class = _load_strategy_class(Path(checkout_root))
        bundle_dir = (
            Path(checkout_root).resolve()
            / OPENSQUILLA_BUNDLE_RELATIVE_PATH
        )
        strategy = strategy_class(
            bundle_dir=bundle_dir,
            require_router_runtime=True,
            emit_train_features=True,
        )
        feature_schema_version = getattr(
            strategy,
            "_feature_schema_version",
            "unknown",
        )
        return cls(
            strategy=strategy,
            bundle_sha256=readiness.bundle_sha256,
            feature_schema_version=feature_schema_version,
        )

    def capture(
        self,
        turn: OpenSquillaTurnInput,
    ) -> tuple[float, ...]:
        request, core, _ = _strategy_runtime_handles(
            self._strategy,
            turn,
        )
        result = core.predict(request)
        intermediates = getattr(result, "intermediates", None)
        if not isinstance(intermediates, Mapping):
            raise RuntimeError("opensquilla_fit_intermediates_missing")
        features = intermediates.get("features_390")
        if features is None:
            raise RuntimeError("opensquilla_fit_features_not_emitted")
        return _validated_feature_vector(features)


class OpenSquillaUpstreamAdapter:
    """Thin wrapper around the pinned strategy's feature/core/postprocess path."""

    def __init__(
        self,
        *,
        strategy: Any,
        bundle_sha256: str,
        projection: OpenSquillaDevProjection,
    ) -> None:
        if not isinstance(projection, OpenSquillaDevProjection):
            raise TypeError("opensquilla_projection_required")
        self._strategy = strategy
        self._bundle_sha256 = _required_sha256(
            bundle_sha256,
            "opensquilla_bundle_sha256",
        )
        self._projection = projection

    @classmethod
    def load(
        cls,
        *,
        checkout_root: str | Path,
        projection: OpenSquillaDevProjection,
        bundle_dir: str | Path | None = None,
        expected_bundle_sha256: str | None = None,
    ) -> OpenSquillaUpstreamAdapter:
        readiness = inspect_opensquilla_readiness(checkout_root)
        if not readiness.ready or readiness.bundle_sha256 is None:
            raise OpenSquillaNotReadyError(readiness)
        strategy_class = _load_strategy_class(Path(checkout_root))
        if bundle_dir is None:
            resolved_bundle = (
                Path(checkout_root).resolve()
                / OPENSQUILLA_BUNDLE_RELATIVE_PATH
            )
            bundle_sha256 = readiness.bundle_sha256
        else:
            resolved_bundle = Path(bundle_dir).resolve()
            blockers = _bundle_blockers(resolved_bundle)
            if blockers:
                detail = "; ".join(
                    f"{blocker.code}: {blocker.detail}"
                    for blocker in blockers
                )
                raise RuntimeError(
                    f"opensquilla_runtime_bundle_not_ready: {detail}"
                )
            bundle_sha256 = _tree_sha256(resolved_bundle)
        if (
            expected_bundle_sha256 is not None
            and bundle_sha256
            != _required_sha256(
                expected_bundle_sha256,
                "opensquilla_expected_bundle_sha256",
            )
        ):
            raise ValueError("opensquilla_runtime_bundle_digest_mismatch")
        strategy = strategy_class(
            bundle_dir=resolved_bundle,
            require_router_runtime=True,
        )
        return cls(
            strategy=strategy,
            bundle_sha256=bundle_sha256,
            projection=projection,
        )

    async def predict(
        self,
        turn: OpenSquillaTurnInput,
    ) -> OpenSquillaPrediction:
        if not isinstance(turn, OpenSquillaTurnInput):
            raise TypeError("opensquilla_turn_input_required")
        request, core, map_result = _strategy_runtime_handles(
            self._strategy,
            turn,
        )
        result = core.predict(request)
        tier, confidence, source, extra = map_result(
            result,
            list(OPENSQUILLA_VALID_TIERS),
            turn.current_user_text,
        )
        if tier not in OPENSQUILLA_VALID_TIERS:
            raise RuntimeError("opensquilla_upstream_tier_invalid")
        if source != "v4_phase3":
            raise RuntimeError("opensquilla_upstream_source_not_v4_phase3")
        if not isinstance(extra, Mapping):
            raise RuntimeError("opensquilla_upstream_extra_missing")
        route_class = str(extra.get("route_class", ""))
        if route_class not in OPENSQUILLA_ROUTE_CLASSES:
            raise RuntimeError("opensquilla_upstream_route_class_invalid")
        resolved_confidence = _unit_float(
            confidence,
            "opensquilla_upstream_confidence",
        )
        probabilities = _validated_probabilities(
            extra.get("probabilities"),
        )
        difficulty = _finite_float(
            extra.get("difficulty"),
            "opensquilla_upstream_difficulty",
        )
        margin = _finite_float(
            extra.get("margin"),
            "opensquilla_upstream_margin",
        )
        typed_route_class: RouteClass = route_class  # type: ignore[assignment]
        return OpenSquillaPrediction(
            route_class=typed_route_class,
            selected_role=self._projection.selected_role(
                typed_route_class,
            ),
            tier=tier,
            confidence=resolved_confidence,
            probabilities=probabilities,
            difficulty=difficulty,
            margin=margin,
            source=source,
            public_input_sha256=turn.sha256,
            bundle_sha256=self._bundle_sha256,
            serializer_config_sha256=(
                OPENSQUILLA_SERIALIZER_CONFIG_SHA256
            ),
            projection_sha256=self._projection.sha256,
        )


def capture_opensquilla_fit_corpus(
    episodes: Sequence[OpenSquillaPairedFitEpisode],
    *,
    fit_manifest: SplitManifest,
    capture: OpenSquillaFeatureCapture,
) -> OpenSquillaFitCorpus:
    """Capture exact upstream features from paired fit-only episode logs."""
    if isinstance(episodes, (str, bytes)) or not isinstance(
        episodes,
        Sequence,
    ):
        raise TypeError("opensquilla_fit_episodes_must_be_sequence")
    if not isinstance(fit_manifest, SplitManifest):
        raise TypeError("opensquilla_fit_manifest_invalid")
    if fit_manifest.role != "router_fit":
        raise ValueError("opensquilla_fit_manifest_role_invalid")
    supplied = tuple(episodes)
    if not supplied or any(
        not isinstance(item, OpenSquillaPairedFitEpisode)
        for item in supplied
    ):
        raise ValueError("opensquilla_fit_episodes_invalid")
    identifiers = tuple(item.example_sha256 for item in supplied)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("opensquilla_fit_episode_duplicate")
    _require_exact_membership(
        fit_manifest,
        identifiers,
        field_name="opensquilla_fit",
    )
    bundle_sha256 = _required_sha256(
        getattr(capture, "bundle_sha256", None),
        "opensquilla_capture_bundle_sha256",
    )
    feature_schema_version = getattr(
        capture,
        "feature_schema_version",
        None,
    )
    if (
        not isinstance(feature_schema_version, str)
        or not feature_schema_version
        or feature_schema_version == "unknown"
    ):
        raise ValueError("opensquilla_capture_feature_schema_invalid")

    by_example = {item.example_sha256: item for item in supplied}
    ordered = tuple(
        by_example[example_sha256]
        for example_sha256 in fit_manifest.example_sha256s
    )
    rows: list[OpenSquillaCapturedFitRow] = []
    for episode in ordered:
        trace_count = len(episode.cheap_turns) + len(
            episode.strong_turns
        )
        per_turn_weight = 1.0 / trace_count
        for source_role, turns, receipt_sha256 in (
            (
                "cheap",
                episode.cheap_turns,
                episode.cheap_episode_receipt_sha256,
            ),
            (
                "strong",
                episode.strong_turns,
                episode.strong_episode_receipt_sha256,
            ),
        ):
            for turn_index, turn in enumerate(turns):
                features = _validated_feature_vector(
                    capture.capture(turn)
                )
                rows.append(
                    OpenSquillaCapturedFitRow(
                        example_sha256=episode.example_sha256,
                        source_role=source_role,  # type: ignore[arg-type]
                        source_episode_receipt_sha256=receipt_sha256,
                        turn_index=turn_index,
                        public_input_sha256=turn.sha256,
                        feature_sha256=_feature_sha256(features),
                        target_route_index=episode.target_route_index,
                        sample_weight=per_turn_weight,
                        features_390=features,
                    )
                )

    return OpenSquillaFitCorpus(
        fit_registry_sha256=fit_manifest.registry_sha256,
        fit_membership_sha256=_membership_sha256(
            fit_manifest,
            tuple(item.example_sha256 for item in ordered),
        ),
        base_bundle_sha256=bundle_sha256,
        feature_schema_version=feature_schema_version,
        source_episode_sha256=hashlib.sha256(
            _canonical_bytes(
                [item.source_record() for item in ordered]
            )
        ).hexdigest(),
        fit_example_sha256s=fit_manifest.example_sha256s,
        rows=tuple(rows),
        paired_fit_log_count=len(ordered),
    )


def train_and_export_opensquilla_fit_head(
    *,
    surface: str,
    fit_manifest: SplitManifest,
    fit_corpus: OpenSquillaFitCorpus,
    checkout_root: str | Path,
    output_dir: str | Path,
) -> OpenSquillaTrainingExport:
    """Continue the pinned upstream head from fit rows and export a full bundle."""
    if not isinstance(surface, str) or not surface.strip():
        raise ValueError("opensquilla_training_surface_invalid")
    if not isinstance(fit_manifest, SplitManifest):
        raise TypeError("opensquilla_fit_manifest_invalid")
    if fit_manifest.role != "router_fit":
        raise ValueError("opensquilla_fit_manifest_role_invalid")
    if not isinstance(fit_corpus, OpenSquillaFitCorpus):
        raise TypeError("opensquilla_fit_corpus_required")
    _require_exact_membership(
        fit_manifest,
        fit_corpus.fit_example_sha256s,
        field_name="opensquilla_training_fit",
    )
    if (
        fit_corpus.fit_registry_sha256 != fit_manifest.registry_sha256
        or fit_corpus.fit_membership_sha256
        != _membership_sha256(
            fit_manifest,
            fit_manifest.example_sha256s,
        )
    ):
        raise ValueError("opensquilla_training_fit_registry_mismatch")

    checkout = Path(checkout_root).resolve()
    readiness = inspect_opensquilla_readiness(checkout)
    if not readiness.ready or readiness.bundle_sha256 is None:
        raise OpenSquillaNotReadyError(readiness)
    if fit_corpus.base_bundle_sha256 != readiness.bundle_sha256:
        raise ValueError("opensquilla_training_base_bundle_mismatch")
    observed_sources = _validated_training_sources(checkout)

    target = _prepare_training_output_dir(Path(output_dir))
    base_bundle = checkout / OPENSQUILLA_BUNDLE_RELATIVE_PATH
    base_lgbm = base_bundle / "lgbm_main.bin"
    if not base_lgbm.is_file():
        raise FileNotFoundError("opensquilla_base_lgbm_missing")

    numpy = importlib.import_module("numpy")
    numpy_version = str(getattr(numpy, "__version__", "unknown"))
    if numpy_version != OPENSQUILLA_TRAINING_NUMPY_VERSION:
        raise RuntimeError(
            "opensquilla_training_numpy_version_mismatch:"
            f"expected={OPENSQUILLA_TRAINING_NUMPY_VERSION},"
            f"found={numpy_version}"
        )
    scikit_learn = importlib.import_module("sklearn")
    lightgbm = importlib.import_module("lightgbm")
    training_module = _load_training_module(checkout)
    x = numpy.asarray(
        [row.features_390 for row in fit_corpus.rows],
        dtype=numpy.float32,
    )
    y = numpy.asarray(
        [row.target_route_index for row in fit_corpus.rows],
        dtype=numpy.int64,
    )
    weights = numpy.asarray(
        [row.sample_weight for row in fit_corpus.rows],
        dtype=numpy.float32,
    )
    config = SimpleNamespace(
        learning_rate=OPENSQUILLA_TRAINING_CONFIG.learning_rate,
        num_leaves=OPENSQUILLA_TRAINING_CONFIG.num_leaves,
        min_data_in_leaf=(
            OPENSQUILLA_TRAINING_CONFIG.min_data_in_leaf
        ),
        num_boost_round=(
            OPENSQUILLA_TRAINING_CONFIG.num_boost_round
        ),
    )
    train_booster_arrays = getattr(
        training_module,
        "train_booster_arrays",
        None,
    )
    assemble_bundle = getattr(
        training_module,
        "assemble_bundle",
        None,
    )
    if not callable(train_booster_arrays) or not callable(assemble_bundle):
        raise RuntimeError("opensquilla_training_surface_missing")
    booster, used_init_model = train_booster_arrays(
        x,
        y,
        weights,
        base_model_path=base_lgbm,
        config=config,
    )
    if used_init_model is not True:
        raise RuntimeError("opensquilla_training_init_model_not_used")

    temporary_head = target / "lgbm_main.bin.tmp"
    save_model = getattr(booster, "save_model", None)
    if not callable(save_model):
        raise TypeError("opensquilla_training_booster_invalid")
    save_model(str(temporary_head))
    bundle_dir = target / OPENSQUILLA_FIT_BUNDLE_RELATIVE_PATH
    assemble_bundle(base_bundle, bundle_dir, temporary_head)
    temporary_head.unlink(missing_ok=True)
    _materialize_bundle_symlinks(bundle_dir)
    blockers = _bundle_blockers(bundle_dir)
    if blockers:
        detail = "; ".join(
            f"{blocker.code}: {blocker.detail}" for blocker in blockers
        )
        raise RuntimeError(
            f"opensquilla_training_bundle_invalid: {detail}"
        )
    adapted_lgbm = bundle_dir / "lgbm_main.bin"
    export = OpenSquillaTrainingExport(
        surface=surface.strip(),
        fit_registry_sha256=fit_manifest.registry_sha256,
        fit_membership_sha256=fit_corpus.fit_membership_sha256,
        fit_corpus_sha256=fit_corpus.sha256,
        source_episode_sha256=fit_corpus.source_episode_sha256,
        fit_example_sha256s=fit_manifest.example_sha256s,
        in_domain_fit_log_count=fit_corpus.paired_fit_log_count,
        fit_turn_count=len(fit_corpus.rows),
        feature_schema_version=fit_corpus.feature_schema_version,
        base_bundle_sha256=readiness.bundle_sha256,
        adapted_bundle_sha256=_tree_sha256(bundle_dir),
        base_lgbm_sha256=_file_sha256(base_lgbm),
        adapted_lgbm_sha256=_file_sha256(adapted_lgbm),
        training_config_sha256=OPENSQUILLA_TRAINING_CONFIG.sha256,
        trainer_source_sha256=OPENSQUILLA_TRAINER_SOURCE_SHA256,
        upstream_source_sha256s=tuple(sorted(observed_sources.items())),
        numpy_version=numpy_version,
        scikit_learn_version=str(
            getattr(scikit_learn, "__version__", "unknown")
        ),
        lightgbm_version=str(getattr(lightgbm, "__version__", "unknown")),
        class_distribution=tuple(
            sorted(fit_corpus.class_distribution.items())
        ),
        used_init_model=True,
    )
    (target / "training_export.json").write_text(
        json.dumps(
            export.json_record(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return export


def load_opensquilla_training_export(
    path: str | Path,
) -> OpenSquillaTrainingExport:
    """Load a fit receipt and re-hash its adapted bundle and model head."""
    export_path = Path(path).resolve()
    raw = json.loads(export_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("opensquilla_training_export_must_be_mapping")
    claimed_sha256 = raw.pop("training_export_sha256", None)
    for field_name in (
        "fit_example_sha256s",
        "upstream_source_sha256s",
        "class_distribution",
    ):
        if not isinstance(raw.get(field_name), list):
            raise ValueError(
                f"opensquilla_training_export_{field_name}_invalid"
            )
    raw["fit_example_sha256s"] = tuple(raw["fit_example_sha256s"])
    raw["upstream_source_sha256s"] = tuple(
        tuple(item) for item in raw["upstream_source_sha256s"]
    )
    raw["class_distribution"] = tuple(
        tuple(item) for item in raw["class_distribution"]
    )
    try:
        export = OpenSquillaTrainingExport(**raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("opensquilla_training_export_schema_invalid") from exc
    if claimed_sha256 != export.sha256:
        raise ValueError("opensquilla_training_export_digest_mismatch")
    bundle_dir = export_path.parent / export.bundle_relative_path
    if not bundle_dir.is_dir():
        raise FileNotFoundError("opensquilla_training_bundle_missing")
    if _tree_sha256(bundle_dir) != export.adapted_bundle_sha256:
        raise ValueError("opensquilla_training_bundle_digest_mismatch")
    if _file_sha256(bundle_dir / "lgbm_main.bin") != (
        export.adapted_lgbm_sha256
    ):
        raise ValueError("opensquilla_training_head_digest_mismatch")
    return export


def inspect_opensquilla_readiness(
    checkout_root: str | Path,
) -> OpenSquillaReadiness:
    root = Path(checkout_root).resolve()
    blockers: list[OpenSquillaReadyBlocker] = []
    if not root.is_dir():
        blockers.append(
            OpenSquillaReadyBlocker(
                "checkout_missing",
                f"pinned checkout directory does not exist: {root}",
            )
        )
        return OpenSquillaReadiness(
            checkout_root=str(root),
            revision=None,
            bundle_sha256=None,
            serializer_config_sha256=(
                OPENSQUILLA_SERIALIZER_CONFIG_SHA256
            ),
            blockers=tuple(blockers),
        )

    revision = _read_git_revision(root)
    if revision is None:
        blockers.append(
            OpenSquillaReadyBlocker(
                "upstream_revision_unreadable",
                f"cannot read git HEAD under {root}",
            )
        )
    elif revision != OPENSQUILLA_UPSTREAM_REVISION:
        blockers.append(
            OpenSquillaReadyBlocker(
                "upstream_revision_mismatch",
                (
                    f"expected {OPENSQUILLA_UPSTREAM_REVISION}, "
                    f"found {revision}"
                ),
            )
        )

    strategy_path = root / OPENSQUILLA_STRATEGY_RELATIVE_PATH
    if not strategy_path.is_file():
        blockers.append(
            OpenSquillaReadyBlocker(
                "strategy_source_missing",
                str(strategy_path),
            )
        )

    bundle_dir = root / OPENSQUILLA_BUNDLE_RELATIVE_PATH
    bundle_sha256: str | None = None
    if not bundle_dir.is_dir():
        blockers.append(
            OpenSquillaReadyBlocker(
                "formal_bundle_missing",
                str(bundle_dir),
            )
        )
    else:
        blockers.extend(_bundle_blockers(bundle_dir))
        bundle_sha256 = _tree_sha256(bundle_dir)

    if _tracked_runtime_dirty(root):
        blockers.append(
            OpenSquillaReadyBlocker(
                "pinned_runtime_dirty",
                (
                    f"{OPENSQUILLA_STRATEGY_RELATIVE_PATH} or "
                    f"{OPENSQUILLA_BUNDLE_RELATIVE_PATH} differs from HEAD"
                ),
            )
        )

    for module_name in _RUNTIME_DEPENDENCIES:
        try:
            available = importlib.util.find_spec(module_name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            available = False
        if not available:
            blockers.append(
                OpenSquillaReadyBlocker(
                    f"missing_python_dependency:{module_name}",
                    (
                        f"module {module_name!r} is required by "
                        "V4Phase3Strategy(require_router_runtime=True)"
                    ),
                )
            )

    return OpenSquillaReadiness(
        checkout_root=str(root),
        revision=revision,
        bundle_sha256=bundle_sha256,
        serializer_config_sha256=OPENSQUILLA_SERIALIZER_CONFIG_SHA256,
        blockers=tuple(blockers),
    )


@dataclass(frozen=True)
class OpenSquillaRuntimeArtifact:
    surface: str
    bundle_sha256: str
    projection: OpenSquillaDevProjection
    fit_registry_sha256: str
    dev_registry_sha256: str
    test_registry_sha256: str
    fit_registry_binding_sha256: str
    dev_membership_sha256: str
    split_binding_sha256: str
    fit_corpus_sha256: str
    training_export_sha256: str
    trainer_config_sha256: str
    trainer_source_sha256: str
    in_domain_fit_log_count: int
    in_domain_fit_turn_count: int
    upstream_revision: str = OPENSQUILLA_UPSTREAM_REVISION
    serializer_config_sha256: str = OPENSQUILLA_SERIALIZER_CONFIG_SHA256
    cheap_model: str = CHEAP_MODEL
    strong_model: str = STRONG_MODEL
    fit_consumption: str = OPENSQUILLA_FIT_CONSUMPTION

    def __post_init__(self) -> None:
        if not isinstance(self.surface, str) or not self.surface.strip():
            raise ValueError("opensquilla_artifact_surface_invalid")
        for field_name in (
            "bundle_sha256",
            "fit_registry_sha256",
            "dev_registry_sha256",
            "test_registry_sha256",
            "fit_registry_binding_sha256",
            "dev_membership_sha256",
            "split_binding_sha256",
            "fit_corpus_sha256",
            "training_export_sha256",
            "trainer_config_sha256",
            "trainer_source_sha256",
        ):
            _required_sha256(getattr(self, field_name), field_name)
        if len(
            {
                self.fit_registry_sha256,
                self.dev_registry_sha256,
                self.test_registry_sha256,
            }
        ) != 3:
            raise ValueError("opensquilla_artifact_split_registry_reused")
        if not isinstance(self.projection, OpenSquillaDevProjection):
            raise TypeError("opensquilla_artifact_projection_invalid")
        if (
            self.projection.dev_registry_sha256
            != self.dev_registry_sha256
            or self.projection.dev_membership_sha256
            != self.dev_membership_sha256
        ):
            raise ValueError("opensquilla_artifact_projection_registry_mismatch")
        if self.upstream_revision != OPENSQUILLA_UPSTREAM_REVISION:
            raise ValueError("opensquilla_artifact_revision_mismatch")
        if (
            self.serializer_config_sha256
            != OPENSQUILLA_SERIALIZER_CONFIG_SHA256
        ):
            raise ValueError("opensquilla_artifact_serializer_mismatch")
        if (self.cheap_model, self.strong_model) != (
            CHEAP_MODEL,
            STRONG_MODEL,
        ):
            raise ValueError("opensquilla_artifact_model_pair_mismatch")
        if self.fit_consumption != OPENSQUILLA_FIT_CONSUMPTION:
            raise ValueError("opensquilla_artifact_fit_consumption_invalid")
        if self.trainer_config_sha256 != (
            OPENSQUILLA_TRAINING_CONFIG.sha256
        ):
            raise ValueError("opensquilla_artifact_trainer_config_mismatch")
        if self.trainer_source_sha256 != (
            OPENSQUILLA_TRAINER_SOURCE_SHA256
        ):
            raise ValueError("opensquilla_artifact_trainer_source_mismatch")
        for field_name in (
            "in_domain_fit_log_count",
            "in_domain_fit_turn_count",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(
                    f"opensquilla_artifact_{field_name}_invalid"
                )
        if self.in_domain_fit_turn_count < self.in_domain_fit_log_count:
            raise ValueError("opensquilla_artifact_fit_turn_count_too_small")

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(
            _canonical_bytes(self.public_record())
        ).hexdigest()

    def public_record(self) -> dict[str, Any]:
        return {
            "schema": OPENSQUILLA_RUNTIME_ARTIFACT_SCHEMA,
            "surface": self.surface,
            "bundle_sha256": self.bundle_sha256,
            "projection": self.projection.public_record(),
            "projection_sha256": self.projection.sha256,
            "fit_registry_sha256": self.fit_registry_sha256,
            "dev_registry_sha256": self.dev_registry_sha256,
            "test_registry_sha256": self.test_registry_sha256,
            "fit_registry_binding_sha256": (
                self.fit_registry_binding_sha256
            ),
            "dev_membership_sha256": self.dev_membership_sha256,
            "split_binding_sha256": self.split_binding_sha256,
            "fit_corpus_sha256": self.fit_corpus_sha256,
            "training_export_sha256": self.training_export_sha256,
            "trainer_config_sha256": self.trainer_config_sha256,
            "trainer_source_sha256": self.trainer_source_sha256,
            "in_domain_fit_log_count": self.in_domain_fit_log_count,
            "in_domain_fit_turn_count": self.in_domain_fit_turn_count,
            "upstream_revision": self.upstream_revision,
            "serializer_config_sha256": self.serializer_config_sha256,
            "cheap_model": self.cheap_model,
            "strong_model": self.strong_model,
            "fit_consumption": self.fit_consumption,
        }

    def router_binding(self) -> RouterArtifactBinding:
        return RouterArtifactBinding(
            method=OPENSQUILLA_G0,
            surface=self.surface,
            upstream_revision=self.upstream_revision,
            artifact_sha256=self.artifact_sha256,
            fit_registry_sha256=self.fit_registry_sha256,
            dev_registry_sha256=self.dev_registry_sha256,
            test_registry_sha256=self.test_registry_sha256,
            calibration_sha256=self.projection.selection_report_sha256,
        )


def bind_opensquilla_runtime_artifact(
    *,
    surface: str,
    training_export: OpenSquillaTrainingExport,
    projection: OpenSquillaDevProjection,
    fit_manifest: SplitManifest,
    dev_manifest: SplitManifest,
    test_manifest: SplitManifest,
) -> OpenSquillaRuntimeArtifact:
    assert_disjoint_manifests(fit_manifest, dev_manifest, test_manifest)
    if not isinstance(training_export, OpenSquillaTrainingExport):
        raise TypeError("opensquilla_training_export_required")
    if training_export.surface != surface:
        raise ValueError("opensquilla_training_export_surface_mismatch")
    if (
        training_export.fit_registry_sha256
        != fit_manifest.registry_sha256
        or training_export.fit_membership_sha256
        != _membership_sha256(
            fit_manifest,
            fit_manifest.example_sha256s,
        )
        or training_export.fit_example_sha256s
        != fit_manifest.example_sha256s
    ):
        raise ValueError("opensquilla_training_export_fit_mismatch")
    if not isinstance(projection, OpenSquillaDevProjection):
        raise TypeError("opensquilla_projection_required")
    if (
        projection.dev_registry_sha256 != dev_manifest.registry_sha256
        or projection.dev_membership_sha256
        != _membership_sha256(dev_manifest, dev_manifest.example_sha256s)
    ):
        raise ValueError("opensquilla_projection_dev_manifest_mismatch")
    return OpenSquillaRuntimeArtifact(
        surface=surface,
        bundle_sha256=_required_sha256(
            training_export.adapted_bundle_sha256,
            "opensquilla_bundle_sha256",
        ),
        projection=projection,
        fit_registry_sha256=fit_manifest.registry_sha256,
        dev_registry_sha256=dev_manifest.registry_sha256,
        test_registry_sha256=test_manifest.registry_sha256,
        fit_registry_binding_sha256=hashlib.sha256(
            _canonical_bytes(
                {
                    "registry_sha256": fit_manifest.registry_sha256,
                    "registry_examples": fit_manifest.example_sha256s,
                    "consumed_examples": (
                        training_export.fit_example_sha256s
                    ),
                    "fit_corpus_sha256": (
                        training_export.fit_corpus_sha256
                    ),
                    "training_export_sha256": training_export.sha256,
                    "fit_consumption": OPENSQUILLA_FIT_CONSUMPTION,
                }
            )
        ).hexdigest(),
        dev_membership_sha256=projection.dev_membership_sha256,
        split_binding_sha256=hashlib.sha256(
            _canonical_bytes(
                {
                    "fit": fit_manifest.__dict__,
                    "dev": dev_manifest.__dict__,
                    "test": test_manifest.__dict__,
                }
            )
        ).hexdigest(),
        fit_corpus_sha256=training_export.fit_corpus_sha256,
        training_export_sha256=training_export.sha256,
        trainer_config_sha256=(
            training_export.training_config_sha256
        ),
        trainer_source_sha256=training_export.trainer_source_sha256,
        in_domain_fit_log_count=(
            training_export.in_domain_fit_log_count
        ),
        in_domain_fit_turn_count=training_export.fit_turn_count,
    )


def opensquilla_artifact_sha256(
    *,
    artifact: OpenSquillaRuntimeArtifact,
) -> str:
    if not isinstance(artifact, OpenSquillaRuntimeArtifact):
        raise TypeError("opensquilla_runtime_artifact_required")
    return artifact.artifact_sha256


def opensquilla_bundle_tree_sha256(bundle_dir: str | Path) -> str:
    """Validate and hash one explicit runtime bundle as an independent asset."""

    resolved = Path(bundle_dir).resolve()
    blockers = _bundle_blockers(resolved)
    if blockers:
        detail = "; ".join(
            f"{blocker.code}: {blocker.detail}" for blocker in blockers
        )
        raise RuntimeError(
            f"opensquilla_runtime_bundle_not_ready: {detail}"
        )
    return _tree_sha256(resolved)


def load_opensquilla_runtime_artifact(
    path: str | Path,
) -> OpenSquillaRuntimeArtifact:
    """Load only the fit-adapted Table 1 artifact schema.

    The old v1 cold-start record is rejected explicitly so it cannot silently
    enter the main-table execution path.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("opensquilla_artifact_record_must_be_mapping")
    schema = raw.pop("schema", None)
    if schema != OPENSQUILLA_RUNTIME_ARTIFACT_SCHEMA:
        if schema == "papc.opensquilla.runtime_artifact.v1":
            raise ValueError("opensquilla_cold_start_artifact_forbidden")
        raise ValueError("opensquilla_artifact_schema_invalid")
    claimed_projection_sha256 = raw.pop("projection_sha256", None)
    projection_record = raw.get("projection")
    if not isinstance(projection_record, Mapping):
        raise ValueError("opensquilla_artifact_projection_invalid")
    try:
        projection = OpenSquillaDevProjection(**dict(projection_record))
        raw["projection"] = projection
        artifact = OpenSquillaRuntimeArtifact(**raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("opensquilla_artifact_schema_invalid") from exc
    if claimed_projection_sha256 != projection.sha256:
        raise ValueError("opensquilla_artifact_projection_digest_mismatch")
    return artifact


def _bundle_blockers(bundle_dir: Path) -> list[OpenSquillaReadyBlocker]:
    blockers: list[OpenSquillaReadyBlocker] = []
    for relative in _REQUIRED_BUNDLE_PATHS:
        path = bundle_dir / relative
        if not path.is_file():
            blockers.append(
                OpenSquillaReadyBlocker(
                    "bundle_file_missing",
                    relative,
                )
            )

    manifest_path = bundle_dir / "artifact_manifest.json"
    if not manifest_path.is_file():
        return blockers
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        blockers.append(
            OpenSquillaReadyBlocker(
                "artifact_manifest_invalid",
                str(exc),
            )
        )
        return blockers

    entries = manifest.get("files")
    if not isinstance(entries, list):
        blockers.append(
            OpenSquillaReadyBlocker(
                "artifact_manifest_invalid",
                "files must be a list",
            )
        )
        return blockers
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            blockers.append(
                OpenSquillaReadyBlocker(
                    "artifact_manifest_entry_invalid",
                    f"index={index}",
                )
            )
            continue
        relative = entry.get("path")
        expected = entry.get("sha256")
        if not isinstance(relative, str) or not relative:
            blockers.append(
                OpenSquillaReadyBlocker(
                    "artifact_manifest_entry_invalid",
                    f"index={index}: path",
                )
            )
            continue
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            blockers.append(
                OpenSquillaReadyBlocker(
                    "artifact_manifest_path_unsafe",
                    relative,
                )
            )
            continue
        path = bundle_dir / relative_path
        if not path.is_file():
            blockers.append(
                OpenSquillaReadyBlocker(
                    "artifact_file_missing",
                    relative,
                )
            )
            continue
        with path.open("rb") as handle:
            if handle.read(48).startswith(
                b"version https://git-lfs.github.com/spec/v1"
            ):
                blockers.append(
                    OpenSquillaReadyBlocker(
                        "artifact_lfs_pointer",
                        (
                            f"{relative}: run git lfs pull for the pinned "
                            "OpenSquilla checkout"
                        ),
                    )
                )
                continue
        expected_size = entry.get("size_bytes")
        if (
            isinstance(expected_size, int)
            and path.stat().st_size != expected_size
        ):
            blockers.append(
                OpenSquillaReadyBlocker(
                    "artifact_size_mismatch",
                    (
                        f"{relative}: expected {expected_size}, "
                        f"found {path.stat().st_size}"
                    ),
                )
            )
            continue
        if not isinstance(expected, str) or not _is_sha256(expected):
            blockers.append(
                OpenSquillaReadyBlocker(
                    "artifact_manifest_sha256_invalid",
                    relative,
                )
            )
            continue
        actual = _file_sha256(path)
        if actual != expected:
            blockers.append(
                OpenSquillaReadyBlocker(
                    "artifact_sha256_mismatch",
                    f"{relative}: expected {expected}, found {actual}",
                )
            )
    return blockers


def _load_strategy_class(checkout_root: Path) -> type[Any]:
    source_root = (checkout_root / "src").resolve()
    loaded = sys.modules.get("opensquilla")
    if loaded is not None:
        module_file = getattr(loaded, "__file__", None)
        if module_file is None or not Path(module_file).resolve().is_relative_to(
            source_root
        ):
            raise RuntimeError("opensquilla_module_origin_conflict")
    sys.path.insert(0, str(source_root))
    try:
        module = importlib.import_module(
            "opensquilla.squilla_router.v4_phase3"
        )
    finally:
        try:
            sys.path.remove(str(source_root))
        except ValueError:
            pass
    strategy_class = getattr(module, "V4Phase3Strategy", None)
    if not isinstance(strategy_class, type):
        raise RuntimeError("opensquilla_strategy_class_missing")
    return strategy_class


def _load_training_module(checkout_root: Path) -> ModuleType:
    source_root = (checkout_root / "src").resolve()
    loaded = sys.modules.get("opensquilla")
    if loaded is not None:
        module_file = getattr(loaded, "__file__", None)
        if (
            module_file is None
            or not Path(module_file).resolve().is_relative_to(source_root)
        ):
            raise RuntimeError("opensquilla_module_origin_conflict")
    sys.path.insert(0, str(source_root))
    try:
        module = importlib.import_module(
            "opensquilla.squilla_router.self_learning.train"
        )
    finally:
        try:
            sys.path.remove(str(source_root))
        except ValueError:
            pass
    return module


def _strategy_runtime_handles(
    strategy: Any,
    turn: OpenSquillaTurnInput,
) -> tuple[Any, Any, Any]:
    if not isinstance(turn, OpenSquillaTurnInput):
        raise TypeError("opensquilla_turn_input_required")
    build_request = getattr(strategy, "_build_request", None)
    core = getattr(strategy, "_core", None)
    map_result = getattr(strategy, "_map_result", None)
    if not callable(build_request) or core is None or not callable(map_result):
        raise RuntimeError("opensquilla_strategy_runtime_surface_missing")
    routing_history = [
        {
            "final_route_class": item.route_class,
            "difficulty_score": item.difficulty,
            "margin": item.margin,
        }
        for item in turn.prior_route_decisions
    ]
    request = build_request(
        turn.current_user_text,
        routing_history,
        prev_assistant_text=turn.previous_assistant_text,
        prev_assistant_usage=dict(turn.previous_assistant_usage),
        history_user_texts=list(turn.recent_user_turns),
        flags_text_override=turn.current_user_text,
    )
    request.context_metadata = turn.public_context_metadata.public_record()
    return request, core, map_result


def _validated_training_sources(checkout_root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative_path, expected_sha256 in sorted(
        OPENSQUILLA_TRAINING_SOURCE_SHA256S.items()
    ):
        path = checkout_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"opensquilla_training_source_missing:{relative_path}"
            )
        actual_sha256 = _file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"opensquilla_training_source_mismatch:{relative_path}"
            )
        observed[relative_path] = actual_sha256
    return observed


def _prepare_training_output_dir(path: Path) -> Path:
    target = path.resolve()
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("opensquilla_training_output_dir_not_empty")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _materialize_bundle_symlinks(bundle_dir: Path) -> None:
    """Turn upstream assembler symlinks into a self-contained bundle."""
    for path in sorted(bundle_dir.iterdir(), key=lambda item: item.name):
        if not path.is_symlink():
            continue
        source = path.resolve(strict=True)
        source_is_dir = source.is_dir()
        path.unlink()
        if source_is_dir:
            shutil.copytree(
                source,
                path,
                ignore=shutil.ignore_patterns(
                    "__pycache__",
                    "*.pyc",
                    "*.pyo",
                ),
            )
        else:
            shutil.copy2(source, path)
    if any(path.is_symlink() for path in bundle_dir.rglob("*")):
        raise ValueError("opensquilla_training_bundle_symlink_remained")


def _validated_feature_vector(values: Any) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError("opensquilla_features_390_invalid")
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError("opensquilla_features_390_invalid") from exc
    if len(raw) != OPENSQUILLA_FEATURE_DIM:
        raise ValueError("opensquilla_features_390_dimension_mismatch")
    resolved: list[float] = []
    for index, value in enumerate(raw):
        if isinstance(value, bool):
            raise TypeError(f"opensquilla_feature_{index}_must_be_numeric")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"opensquilla_feature_{index}_must_be_numeric"
            ) from exc
        if not math.isfinite(numeric):
            raise ValueError(f"opensquilla_feature_{index}_not_finite")
        resolved.append(numeric)
    return tuple(resolved)


def _feature_sha256(features: tuple[float, ...]) -> str:
    resolved = _validated_feature_vector(features)
    return hashlib.sha256(
        struct.pack(f"<{OPENSQUILLA_FEATURE_DIM}f", *resolved)
    ).hexdigest()


def _read_git_revision(root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    revision = completed.stdout.strip()
    return revision if len(revision) == 40 else None


def _tracked_runtime_dirty(root: Path) -> bool:
    paths = (
        str(OPENSQUILLA_STRATEGY_RELATIVE_PATH),
        str(OPENSQUILLA_BUNDLE_RELATIVE_PATH),
    )
    completed = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--", *paths],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode != 0 or bool(completed.stdout.strip())


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    digest.update(b"papc.opensquilla.bundle-tree.v1\0")
    for path in sorted(
        (
            item
            for item in root.rglob("*")
            if item.is_file()
            and "__pycache__" not in item.relative_to(root).parts
            and item.suffix not in {".pyc", ".pyo"}
        ),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        if path.is_symlink():
            raise ValueError("opensquilla_bundle_symlink_forbidden")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _validated_usage(
    raw: Mapping[str, int | float],
) -> dict[str, int | float]:
    if not isinstance(raw, Mapping):
        raise TypeError("opensquilla_previous_assistant_usage_invalid")
    unknown = sorted(set(raw) - _USAGE_FIELDS)
    if unknown:
        raise ValueError(
            "opensquilla_previous_assistant_usage_unknown:"
            + ",".join(unknown)
        )
    usage: dict[str, int | float] = {}
    for name, value in raw.items():
        resolved = _finite_float(
            value,
            f"opensquilla_previous_assistant_usage_{name}",
        )
        if resolved < 0:
            raise ValueError(
                f"opensquilla_previous_assistant_usage_{name}_negative"
            )
        usage[name] = int(resolved) if resolved.is_integer() else resolved
    return usage


def _validated_probabilities(
    raw: Any,
) -> tuple[tuple[str, float], ...]:
    if not isinstance(raw, Mapping):
        raise RuntimeError("opensquilla_upstream_probabilities_missing")
    if set(raw) != set(OPENSQUILLA_ROUTE_CLASSES):
        raise RuntimeError("opensquilla_upstream_probability_classes_invalid")
    probabilities = tuple(
        (
            route_class,
            _unit_float(
                raw[route_class],
                f"opensquilla_upstream_probability_{route_class}",
            ),
        )
        for route_class in OPENSQUILLA_ROUTE_CLASSES
    )
    total = sum(value for _, value in probabilities)
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise RuntimeError("opensquilla_upstream_probabilities_not_normalized")
    return probabilities


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}_invalid")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name}_must_be_numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name}_not_finite")
    return resolved


def _unit_float(value: Any, name: str) -> float:
    resolved = _finite_float(value, name)
    if resolved < 0.0 or resolved > 1.0:
        raise ValueError(f"{name}_outside_unit_interval")
    return resolved


def _required_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _is_sha256(value):
        raise ValueError(f"{name}_invalid")
    return value


def _projection_config_sha256(strong_class_floor: int) -> str:
    if (
        isinstance(strong_class_floor, bool)
        or strong_class_floor not in {0, 1, 2, 3, 4}
    ):
        raise ValueError("opensquilla_projection_floor_invalid")
    return hashlib.sha256(
        _canonical_bytes(
            {
                "method": OPENSQUILLA_G0,
                "projection": "monotone_route_class_floor",
                "strong_class_floor": strong_class_floor,
                "route_classes": OPENSQUILLA_ROUTE_CLASSES,
                "cheap_model": CHEAP_MODEL,
                "strong_model": STRONG_MODEL,
            }
        )
    ).hexdigest()


def _require_exact_membership(
    manifest: SplitManifest,
    example_sha256s: tuple[str, ...],
    *,
    field_name: str,
) -> None:
    if len(example_sha256s) != len(set(example_sha256s)):
        raise ValueError(f"{field_name}_example_duplicate")
    if set(example_sha256s) != set(manifest.example_sha256s):
        raise ValueError(f"{field_name}_registry_membership_mismatch")


def _membership_sha256(
    manifest: SplitManifest,
    example_sha256s: tuple[str, ...],
) -> str:
    _require_exact_membership(
        manifest,
        example_sha256s,
        field_name="opensquilla_membership",
    )
    return hashlib.sha256(
        _canonical_bytes(
            {
                "role": manifest.role,
                "registry_sha256": manifest.registry_sha256,
                "example_sha256s": manifest.example_sha256s,
            }
        )
    ).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

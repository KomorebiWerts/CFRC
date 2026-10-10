"""Small, source-faithful StepWise fit adaptation for Agent-Diff.

The module mirrors the pinned WebRouter annotation schemas and sliding-window
datasets.  It contains no provider client and does not read credentials; the
paid annotation script injects the provider boundary and stores receipts.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from papc.baselines.stepwise_upstream import (
    STEPWISE_CONTEXT_WINDOW,
    StepWisePublicStep,
    public_steps,
)

STEPWISE_FIT_SCHEMA = "papc.stepwise_agent_diff_fit.v1"
STEPWISE_ANNOTATION_MODEL_MAX_TOKENS = 2000
STEPWISE_TRAIN_TEST_FRACTION = 0.2
STEPWISE_TRAIN_TEST_SEED = 42

AnnotationKind = Literal["stuck", "milestones"]


@dataclass(frozen=True)
class StepWiseFitTrajectory:
    """One outcome-blind cheap trajectory selected from the fit registry."""

    example_sha256: str
    task_name: str
    original_request: str
    public_episode: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        _require_sha256(self.example_sha256, "stepwise_fit_example_sha256")
        for name in ("task_name", "original_request"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"stepwise_fit_{name}_invalid")
        episode = tuple(dict(item) for item in self.public_episode)
        if not episode or not public_steps(episode):
            raise ValueError("stepwise_fit_public_episode_has_no_agent_step")
        object.__setattr__(self, "public_episode", episode)

    @property
    def steps(self) -> tuple[StepWisePublicStep, ...]:
        return public_steps(self.public_episode)


@dataclass(frozen=True)
class StepWiseAnnotationRequest:
    """One exact structured-output request for a StepWise fit trajectory."""

    kind: AnnotationKind
    messages: tuple[Mapping[str, str], ...]
    response_format: Mapping[str, Any]
    max_tokens: int = STEPWISE_ANNOTATION_MODEL_MAX_TOKENS

    def __post_init__(self) -> None:
        if self.kind not in {"stuck", "milestones"}:
            raise ValueError("stepwise_annotation_kind_invalid")
        messages = tuple(dict(item) for item in self.messages)
        if len(messages) != 2 or tuple(item.get("role") for item in messages) != (
            "system",
            "user",
        ):
            raise ValueError("stepwise_annotation_messages_invalid")
        if any(
            not isinstance(item.get("content"), str)
            or not item["content"].strip()
            for item in messages
        ):
            raise ValueError("stepwise_annotation_message_content_invalid")
        response_format = dict(self.response_format)
        if response_format.get("type") != "json_schema":
            raise ValueError("stepwise_annotation_json_schema_required")
        if self.max_tokens != STEPWISE_ANNOTATION_MODEL_MAX_TOKENS:
            raise ValueError("stepwise_annotation_max_tokens_not_frozen")
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "response_format", response_format)

    @property
    def sha256(self) -> str:
        return _digest(
            {
                "kind": self.kind,
                "messages": self.messages,
                "response_format": self.response_format,
                "max_tokens": self.max_tokens,
            }
        )

    def public_record(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "request_sha256": self.sha256,
            "max_tokens": self.max_tokens,
            "message_count": len(self.messages),
            "response_format": "strict_json_schema",
        }


@dataclass(frozen=True)
class StepWiseAnnotation:
    """Validated positive step labels from one structured annotation call."""

    kind: AnnotationKind
    positive_steps: tuple[int, ...]
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.kind not in {"stuck", "milestones"}:
            raise ValueError("stepwise_annotation_kind_invalid")
        steps = tuple(self.positive_steps)
        if steps != tuple(sorted(set(steps))) or any(
            isinstance(step, bool) or not isinstance(step, int) or step < 0
            for step in steps
        ):
            raise ValueError("stepwise_annotation_positive_steps_invalid")
        object.__setattr__(self, "positive_steps", steps)
        object.__setattr__(self, "payload", dict(self.payload))


def build_annotation_request(
    trajectory: StepWiseFitTrajectory,
    *,
    kind: AnnotationKind,
) -> StepWiseAnnotationRequest:
    """Translate one public Agent-Diff episode into the pinned annotation task."""

    if not isinstance(trajectory, StepWiseFitTrajectory):
        raise TypeError("stepwise_fit_trajectory_required")
    formatted = _format_trajectory(trajectory.steps)
    goal = trajectory.original_request[:500]
    if kind == "stuck":
        system = (
            "You are an expert AI agent evaluator. Your task is to analyze "
            "agent trajectories and identify if the agent got stuck during "
            "execution.\n\n"
            'An agent is considered "stuck" if:\n'
            "1. It repeats the same action multiple times without progress\n"
            "2. It enters an error loop or infinite loop\n"
            "3. It failed to make meaningful progress for several steps\n\n"
            "Analyze the trajectory and return the requested JSON object."
        )
        user = (
            f"Analyze the following agent trajectory for task: "
            f"{trajectory.task_name}\n\n"
            f"Goal: {goal}\n\n"
            f"Total steps: {len(trajectory.steps)}\n\n"
            f"Trajectory:\n{formatted}\n\n"
            "Identify stuck steps and explain each label."
        )
        schema = _stuck_response_format()
    elif kind == "milestones":
        system = (
            "You are labeling a tool-agent trajectory to identify milestone "
            "steps -- steps where meaningful, verifiable progress is achieved.\n\n"
            "1. If the trajectory is short/simple, like 5 or 6 steps, you may "
            "output only the final step as a milestone.\n"
            "2. If the trajectory is long, you may list multiple milestones, "
            "but do not list too many, and each two milestones should be at "
            "least 3 steps apart.\n"
            "3. If the trajectory becomes stuck (repetition/no progress), "
            "ignore steps inside the stuck region unless a milestone occurs "
            "later.\n\n"
            "A milestone must be meaningful and verifiable from the given "
            "action/response/done/fail text. Prefer higher-level progress "
            "markers, do not invent unsupported details, and explain each "
            "milestone."
        )
        user = (
            f"Task: {trajectory.task_name}\n\n"
            f"Goal: {goal}\n\n"
            f"Total steps: {len(trajectory.steps)}\n\n"
            f"Trajectory:\n{formatted}\n\n"
            "Return JSON only. For each milestone, provide the step number "
            "and why it is meaningful progress."
        )
        schema = _milestone_response_format()
    else:
        raise ValueError("stepwise_annotation_kind_invalid")
    return StepWiseAnnotationRequest(
        kind=kind,
        messages=(
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ),
        response_format=schema,
    )


def parse_annotation(
    *,
    kind: AnnotationKind,
    payload: Mapping[str, Any],
    step_count: int,
) -> StepWiseAnnotation:
    """Fail closed on malformed or out-of-range structured labels."""

    if isinstance(step_count, bool) or not isinstance(step_count, int) or step_count < 1:
        raise ValueError("stepwise_annotation_step_count_invalid")
    raw = dict(payload)
    if kind == "stuck":
        is_stuck = raw.get("is_stuck")
        steps = _integer_steps(raw.get("stuck_steps"), step_count=step_count)
        _string_list(raw.get("reasons"), "stuck_reasons")
        if not isinstance(is_stuck, bool):
            raise TypeError("stepwise_stuck_is_stuck_invalid")
        # The released StepWise dataset builder treats ``stuck_steps`` as the
        # per-step label authority and does not consult the trajectory-level
        # summary fields. Keep that upstream boundary here while still
        # validating every returned field structurally.
        if raw.get("severity") not in {"none", "low", "medium", "high"}:
            raise ValueError("stepwise_stuck_severity_invalid")
        if not isinstance(raw.get("summary"), str):
            raise TypeError("stepwise_stuck_summary_invalid")
    elif kind == "milestones":
        steps = _integer_steps(
            raw.get("milestone_steps"),
            step_count=step_count,
            maximum=12,
        )
        _string_list(
            raw.get("milestone_reasons"),
            "milestone_reasons",
        )
    else:
        raise ValueError("stepwise_annotation_kind_invalid")
    return StepWiseAnnotation(kind=kind, positive_steps=steps, payload=raw)


def build_training_datasets(
    trajectories: Sequence[StepWiseFitTrajectory],
    annotations: Mapping[tuple[str, AnnotationKind], StepWiseAnnotation],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Build the two upstream-compatible datasets with a task-level split."""

    resolved = tuple(trajectories)
    if len(resolved) < 2:
        raise ValueError("stepwise_fit_at_least_two_trajectories_required")
    if len({item.example_sha256 for item in resolved}) != len(resolved):
        raise ValueError("stepwise_fit_trajectory_duplicate")
    split = _task_split(tuple(item.task_name for item in resolved))
    datasets: dict[str, list[dict[str, Any]]] = {
        "stuck": [],
        "milestones": [],
    }
    for trajectory in resolved:
        for kind in ("stuck", "milestones"):
            annotation = annotations.get((trajectory.example_sha256, kind))
            if not isinstance(annotation, StepWiseAnnotation) or annotation.kind != kind:
                raise ValueError(
                    f"stepwise_fit_annotation_missing:{trajectory.example_sha256}:{kind}"
                )
            positive = set(annotation.positive_steps)
            for index, step in enumerate(trajectory.steps):
                datasets[kind].append(
                    {
                        "text": _training_window(
                            trajectory,
                            current_index=index,
                            include_goal=kind == "milestones",
                        ),
                        "label": int(step.index in positive),
                        "task_name": trajectory.task_name,
                        "step_num": step.index,
                        "split": split[trajectory.task_name],
                    }
                )
    stats = {
        "schema": STEPWISE_FIT_SCHEMA,
        "trajectory_count": len(resolved),
        "task_level_split": True,
        "split_seed": STEPWISE_TRAIN_TEST_SEED,
        "test_fraction": STEPWISE_TRAIN_TEST_FRACTION,
        "datasets": {
            kind: _dataset_stats(rows) for kind, rows in datasets.items()
        },
    }
    stats["ready_for_upstream_training"] = all(
        item["training_labels"] == [0, 1]
        for item in stats["datasets"].values()
    )
    return datasets, stats


def _format_trajectory(steps: Sequence[StepWisePublicStep]) -> str:
    lines: list[str] = []
    for step in steps:
        think = step.think
        if len(think) > 500:
            think = think[:500] + "..."
        suffix = f" {think}" if think else ""
        lines.append(f"Step {step.index}: [{step.action}]{suffix}")
    return "\n".join(lines)


def _training_window(
    trajectory: StepWiseFitTrajectory,
    *,
    current_index: int,
    include_goal: bool,
) -> str:
    steps = trajectory.steps
    start = max(0, current_index - STEPWISE_CONTEXT_WINDOW + 1)
    lines = []
    if include_goal:
        lines.append(f"Task: {trajectory.original_request[:500].strip()}")
    for step in steps[start : current_index + 1]:
        suffix = f" {step.think}" if step.think else ""
        lines.append(f"Step {step.index}: [{step.action}]{suffix}")
    return "\n".join(lines)


def _task_split(task_names: Sequence[str]) -> dict[str, str]:
    names = sorted(set(task_names))
    if len(names) != len(task_names):
        raise ValueError("stepwise_fit_task_name_duplicate")
    random.Random(STEPWISE_TRAIN_TEST_SEED).shuffle(names)
    test_count = max(1, int(len(names) * STEPWISE_TRAIN_TEST_FRACTION))
    test = set(names[:test_count])
    return {name: "test" if name in test else "train" for name in names}


def _dataset_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = Counter(int(row["label"]) for row in rows)
    training = [row for row in rows if row["split"] == "train"]
    training_labels = sorted({int(row["label"]) for row in training})
    return {
        "sample_count": len(rows),
        "task_count": len({str(row["task_name"]) for row in rows}),
        "label_counts": {str(key): value for key, value in sorted(labels.items())},
        "training_labels": training_labels,
        "train_samples": len(training),
        "test_samples": len(rows) - len(training),
    }


def _integer_steps(
    value: Any,
    *,
    step_count: int,
    maximum: int | None = None,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("stepwise_annotation_steps_invalid")
    steps = tuple(value)
    if maximum is not None and len(steps) > maximum:
        raise ValueError("stepwise_annotation_too_many_steps")
    if any(
        isinstance(step, bool)
        or not isinstance(step, int)
        or step < 0
        or step >= step_count
        for step in steps
    ):
        raise ValueError("stepwise_annotation_step_out_of_range")
    if steps != tuple(sorted(set(steps))):
        raise ValueError("stepwise_annotation_steps_not_ordered_unique")
    return steps


def _string_list(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"stepwise_{field_name}_invalid")
    result = tuple(value)
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise ValueError(f"stepwise_{field_name}_invalid")
    return result


def _stuck_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "trajectory_stuck_analysis",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "is_stuck": {"type": "boolean"},
                    "stuck_steps": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    "reasons": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["none", "low", "medium", "high"],
                    },
                    "summary": {"type": "string"},
                },
                "required": [
                    "is_stuck",
                    "stuck_steps",
                    "reasons",
                    "severity",
                    "summary",
                ],
                "additionalProperties": False,
            },
        },
    }


def _milestone_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "trajectory_milestones",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "milestone_steps": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "maxItems": 12,
                    },
                    "milestone_reasons": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 12,
                    },
                },
                "required": ["milestone_steps", "milestone_reasons"],
                "additionalProperties": False,
            },
        },
    }


def _require_sha256(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name}_invalid")
    return value


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
    "STEPWISE_ANNOTATION_MODEL_MAX_TOKENS",
    "STEPWISE_FIT_SCHEMA",
    "STEPWISE_TRAIN_TEST_FRACTION",
    "STEPWISE_TRAIN_TEST_SEED",
    "AnnotationKind",
    "StepWiseAnnotation",
    "StepWiseAnnotationRequest",
    "StepWiseFitTrajectory",
    "build_annotation_request",
    "build_training_datasets",
    "parse_annotation",
]

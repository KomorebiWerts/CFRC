"""Pinned StepWise event-driven escalation semantics for PAPC.

The adaptation follows the Apache-2.0 ``WebRouter`` implementation from
``yale-nlp/StepWise``.  A cheap model acts by default.  Two local monitors read
only the public trajectory: a stuck event escalates immediately, while a
milestone event asks the strong model to verify progress.  Failed verification
causes a permanent takeover from the current episode state.

This module is deliberately provider-neutral.  Monitor inference and strong
verification are injected callbacks so provider attempts remain visible to the
surface runner and its accounting broker.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from papc.baselines.router_contract import HistoryRouterView

STEPWISE_METHOD = "stepwise_event_escalation"
STEPWISE_UPSTREAM_REVISION = "caf2984bd0fc381161b18b019e1570fce0dc3503"
STEPWISE_LICENSE = "Apache-2.0"
STEPWISE_LICENSE_SCOPE = "WebRouter"
STEPWISE_CONTEXT_WINDOW = 5
STEPWISE_STUCK_THRESHOLD = 0.1
STEPWISE_MILESTONE_THRESHOLD = 0.1
STEPWISE_MONITOR_BASE_MODEL = "answerdotai/ModernBERT-base"
STEPWISE_ANNOTATION_CALLS_PER_FIT_TRAJECTORY = 2
STEPWISE_SOURCE_SHA256S = {
    "WebRouter/LICENSE": (
        "b192c58991e8ff585cc574615d40e74185404d4b96c1109d423071ab1367344b"
    ),
    "WebRouter/scripts/bert/switching_agent.py": (
        "955e5e1f5b4610bb324691aaca23fda8a7443126190cb143b1188ce7095f8932"
    ),
    "WebRouter/scripts/bert/annotate_trajectories.py": (
        "f38a027a2a4ffefb58a64acae38846342849dd3f7b5f45a378ef32ddce0e4693"
    ),
    "WebRouter/scripts/bert/build_stuck_dataset.py": (
        "a3c119df47ecc5ab9c73710d8a30967bd723c1febe97a1f32763eb194fe275ac"
    ),
    "WebRouter/scripts/bert/build_milestone_dataset.py": (
        "fe5dcfdd8b3d4e170000395820a9918ce9e9aabaabd22605739de40bbabe9cab"
    ),
    "WebRouter/scripts/bert/train_router.py": (
        "f1239c582737081ed431e8a5a1a361428831378f1d9c5ad8fe898d47c5244d1f"
    ),
}

StepWiseRole = Literal["cheap", "strong"]
StepWiseMonitor = Callable[[str, str], tuple[float, float]]
StepWiseVerifier = Callable[[str], bool]

_REACT_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
_REACT_THINK_RE = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL | re.IGNORECASE)
_REACT_DONE_RE = re.compile(r"<done>(.*?)</done>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class StepWisePublicStep:
    """One assistant action exposed to the StepWise monitors."""

    index: int
    action: str
    think: str


@dataclass(frozen=True)
class StepWiseDecision:
    """One stateful routing decision before a native agent turn."""

    selected_role: StepWiseRole
    rationale: str
    public_input_sha256: str
    stuck_probability: float | None
    milestone_probability: float | None
    milestone_verified: bool | None
    permanently_escalated: bool

    def public_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StepWiseCheckoutReadiness:
    """Provider-zero inspection of the pinned licensed source surface."""

    checkout: str
    observed_revision: str | None
    source_sha256s: tuple[tuple[str, str | None], ...]
    blockers: tuple[str, ...]
    provider_calls: int = 0
    secret_reads: int = 0

    @property
    def ready(self) -> bool:
        return not self.blockers

    def public_record(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "ready": self.ready,
            "expected_revision": STEPWISE_UPSTREAM_REVISION,
            "license": STEPWISE_LICENSE,
            "license_scope": STEPWISE_LICENSE_SCOPE,
        }


class StepWiseController:
    """Faithful permanent-escalation controller with injected local monitors."""

    def __init__(
        self,
        *,
        stuck_threshold: float = STEPWISE_STUCK_THRESHOLD,
        milestone_threshold: float = STEPWISE_MILESTONE_THRESHOLD,
    ) -> None:
        self.stuck_threshold = _unit_interval(
            stuck_threshold,
            "stepwise_stuck_threshold",
        )
        self.milestone_threshold = _unit_interval(
            milestone_threshold,
            "stepwise_milestone_threshold",
        )
        self._using_strong = False
        self._last_verified_step = -1

    @property
    def using_strong(self) -> bool:
        return self._using_strong

    def decide(
        self,
        *,
        view: HistoryRouterView,
        monitor: StepWiseMonitor,
        verifier: StepWiseVerifier,
    ) -> StepWiseDecision:
        """Select the model for the next turn from public history only."""

        if not isinstance(view, HistoryRouterView):
            raise TypeError("stepwise_history_view_required")
        if self._using_strong:
            return self._decision(
                view=view,
                role="strong",
                rationale="permanent_takeover_already_active",
            )

        steps = public_steps(view.public_history)
        if not steps:
            return self._decision(
                view=view,
                role="cheap",
                rationale="initial_cheap_turn",
            )

        stuck_text = monitor_text(view, include_goal=False)
        milestone_text = monitor_text(view, include_goal=True)
        stuck_probability, milestone_probability = monitor(
            stuck_text,
            milestone_text,
        )
        stuck = _unit_interval(
            stuck_probability,
            "stepwise_stuck_probability",
        )
        milestone = _unit_interval(
            milestone_probability,
            "stepwise_milestone_probability",
        )
        if stuck >= self.stuck_threshold:
            self._using_strong = True
            return self._decision(
                view=view,
                role="strong",
                rationale="stuck_event_permanent_takeover",
                stuck=stuck,
                milestone=milestone,
            )

        verified: bool | None = None
        if milestone >= self.milestone_threshold:
            new_steps = tuple(
                step
                for step in steps
                if step.index > self._last_verified_step
            )
            if new_steps:
                verified = verifier(
                    verification_text(
                        view,
                        after_step=self._last_verified_step,
                    )
                )
                if not isinstance(verified, bool):
                    raise TypeError("stepwise_verifier_result_must_be_bool")
                self._last_verified_step = new_steps[-1].index
                if not verified:
                    self._using_strong = True
                    return self._decision(
                        view=view,
                        role="strong",
                        rationale=(
                            "milestone_verification_failed_permanent_takeover"
                        ),
                        stuck=stuck,
                        milestone=milestone,
                        verified=False,
                    )

        return self._decision(
            view=view,
            role="cheap",
            rationale=(
                "milestone_verified_continue_cheap"
                if verified is True
                else (
                    "milestone_without_new_step_continue_cheap"
                    if milestone >= self.milestone_threshold
                    else "no_escalation_event"
                )
            ),
            stuck=stuck,
            milestone=milestone,
            verified=verified,
        )

    def _decision(
        self,
        *,
        view: HistoryRouterView,
        role: StepWiseRole,
        rationale: str,
        stuck: float | None = None,
        milestone: float | None = None,
        verified: bool | None = None,
    ) -> StepWiseDecision:
        return StepWiseDecision(
            selected_role=role,
            rationale=rationale,
            public_input_sha256=view.sha256,
            stuck_probability=stuck,
            milestone_probability=milestone,
            milestone_verified=verified,
            permanently_escalated=self._using_strong,
        )


def public_steps(
    public_history: Sequence[Mapping[str, Any]],
) -> tuple[StepWisePublicStep, ...]:
    """Map public assistant messages to the released action/think format."""

    steps: list[StepWisePublicStep] = []
    for message in public_history:
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        content = message.get("content")
        if isinstance(calls, Sequence) and not isinstance(calls, (str, bytes)):
            actions = tuple(_tool_call_text(call) for call in calls)
            actions = tuple(action for action in actions if action)
        else:
            actions = ()
        think = content if isinstance(content, str) else ""
        if actions:
            action = " | ".join(actions)
        elif think:
            action, think = _assistant_text_step(think)
        else:
            continue
        steps.append(
            StepWisePublicStep(
                index=len(steps),
                action=action,
                think=think,
            )
        )
    return tuple(steps)


def monitor_text(
    view: HistoryRouterView,
    *,
    include_goal: bool,
) -> str:
    """Serialize the same five-step monitor window as released WebRouter."""

    if not isinstance(view, HistoryRouterView):
        raise TypeError("stepwise_history_view_required")
    steps = public_steps(view.public_history)[-STEPWISE_CONTEXT_WINDOW:]
    lines = []
    if include_goal:
        lines.append(f"Task: {_short_goal(view.original_request)}")
    lines.extend(_step_line(step) for step in steps)
    return "\n".join(lines)


def verification_text(
    view: HistoryRouterView,
    *,
    after_step: int,
) -> str:
    """Build the strong milestone-verification packet from public actions."""

    if not isinstance(after_step, int) or isinstance(after_step, bool):
        raise TypeError("stepwise_after_step_must_be_int")
    steps = tuple(
        step
        for step in public_steps(view.public_history)
        if step.index > after_step
    )
    if not steps:
        raise ValueError("stepwise_verification_steps_empty")
    return (
        "Task:\n"
        f"{_short_goal(view.original_request)}\n\n"
        "Steps since the last verified checkpoint:\n"
        + "\n".join(_step_line(step) for step in steps)
        + "\n\nReturn exactly CORRECT or INCORRECT."
    )


def inspect_stepwise_checkout(checkout: Path) -> StepWiseCheckoutReadiness:
    """Verify the exact revision, licensed subtree, and pinned source bytes."""

    root = Path(checkout).resolve()
    blockers: list[str] = []
    observed_revision: str | None = None
    if not root.is_dir():
        blockers.append("stepwise_checkout_missing")
    else:
        try:
            observed_revision = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            blockers.append("stepwise_checkout_revision_unreadable")
    if (
        observed_revision is not None
        and observed_revision != STEPWISE_UPSTREAM_REVISION
    ):
        blockers.append("stepwise_checkout_revision_mismatch")

    observed_hashes: list[tuple[str, str | None]] = []
    for relative, expected in STEPWISE_SOURCE_SHA256S.items():
        path = root / relative
        observed = _file_sha256(path) if path.is_file() else None
        observed_hashes.append((relative, observed))
        if observed != expected:
            blockers.append(f"stepwise_source_mismatch:{relative}")

    license_path = root / "WebRouter/LICENSE"
    if license_path.is_file():
        license_text = license_path.read_text(encoding="utf-8")
        if "Apache License" not in license_text or "Version 2.0" not in license_text:
            blockers.append("stepwise_webrouter_license_not_apache_2_0")

    return StepWiseCheckoutReadiness(
        checkout=str(root),
        observed_revision=observed_revision,
        source_sha256s=tuple(observed_hashes),
        blockers=tuple(dict.fromkeys(blockers)),
    )


def _tool_call_text(call: Any) -> str:
    if not isinstance(call, Mapping):
        return ""
    function = call.get("function")
    source = function if isinstance(function, Mapping) else call
    name = source.get("name")
    if not isinstance(name, str) or not name.strip():
        return ""
    arguments = source.get("arguments", "")
    if isinstance(arguments, str):
        argument_text = arguments
    else:
        argument_text = json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return f"{name.strip()}({argument_text})"


def _assistant_text_step(content: str) -> tuple[str, str]:
    """Recover the public ReAct action/think pair used by Agent-Diff."""

    action = _REACT_ACTION_RE.search(content)
    thought = _REACT_THINK_RE.search(content)
    done = _REACT_DONE_RE.search(content)
    think = thought.group(1).strip() if thought else content.strip()
    if action:
        return action.group(1).strip(), think
    if done:
        return "done", think
    return "message", think


def _step_line(step: StepWisePublicStep) -> str:
    suffix = f" {step.think}" if step.think else ""
    return f"Step {step.index}: [{step.action}]{suffix}"


def _short_goal(value: str) -> str:
    for separator in ("\n---", "\n===", "\nYour response should"):
        if separator in value:
            return value[: value.index(separator)].strip()
    return value[:500].strip()


def _unit_interval(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name}_must_be_numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name}_not_finite")
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{field_name}_outside_unit_interval")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

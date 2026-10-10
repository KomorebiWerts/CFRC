"""Sanitized ToolSandbox residual certificates for offline PAPC gates.

This module builds ``z = C(tau)`` from public-safe runtime digests only.  It does
not consume scenario names, task ids, gold milestones, raw prompts, hidden state
snapshots, fixed expected arguments, or provider responses.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from papc.certspec.toolsandbox.state_write_contract import (
    DEPENDENCY_COMPLETION_UNIT,
    MINEFIELD_SIDE_EFFECT_UNIT,
    OBSERVED_STATE_UNIT_PREFIX,
    SURFACE,
)
from papc.paths import DATA_ROOT


@dataclass(frozen=True)
class ToolSandboxResidualCertificate:
    """Public-safe residual certificate for one ToolSandbox prefix."""

    visible_prefix: Mapping[str, Any] = field(default_factory=dict)
    evidence: tuple[str, ...] = ()
    commitments: tuple[str, ...] = ()
    residual_obligations: tuple[str, ...] = ()
    legal_residual_actions: tuple[str, ...] = ()
    authority: Mapping[str, Any] = field(default_factory=dict)
    forbidden_effects: tuple[str, ...] = ()
    provider_calls: int = 0

    @property
    def activation_ready(self) -> bool:
        return (
            bool(self.residual_obligations)
            and DEPENDENCY_COMPLETION_UNIT in self.legal_residual_actions
            and self.provider_calls == 0
            and bool(self.evidence)
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "papc.toolsandbox.residual_certificate.v1",
            "surface": SURFACE,
            "E(z)": list(self.evidence),
            "K(z)": list(self.commitments),
            "R(z)": list(self.residual_obligations),
            "S(z)": list(self.legal_residual_actions),
            "Omega(z)": dict(self.authority),
            "forbidden_effects": list(self.forbidden_effects),
            "activation_ready": self.activation_ready,
            "provider_calls": self.provider_calls,
            "private_or_hidden_fields_used": False,
        }


def certificate_from_runtime_digest(
    digest: Mapping[str, Any],
) -> ToolSandboxResidualCertificate:
    """Build a residual certificate from a sanitized ToolSandbox digest."""

    visible_prefix = _mapping(digest.get("visible_prefix"))
    observed_state = _mapping(digest.get("observed_state"))
    residual_request = _mapping(digest.get("residual_request"))
    certificate_digest = _mapping(digest.get("certificate_digest"))

    observed_count = _nonnegative_int(observed_state.get("observed_state_units"))
    evidence = tuple(f"{OBSERVED_STATE_UNIT_PREFIX}#{index}" for index in range(observed_count))
    if visible_prefix.get("visible_tool_result_count"):
        evidence += ("visible_tool_results_present",)

    commitments: list[str] = []
    if residual_request.get("requires_illegal_side_effect_avoidance"):
        commitments.append("avoid_minefield_side_effect")
    if residual_request.get("approval_granted") is True:
        commitments.append("approval_granted")
    if residual_request.get("approval_denied") is True:
        commitments.append("approval_denied")
    if residual_request.get("wait_state") is True:
        commitments.append("wait_state_observed")

    residual_obligations: list[str] = []
    legal_actions: list[str] = ["abstain"]
    if residual_request.get("requires_dependency_closure"):
        residual_obligations.append(DEPENDENCY_COMPLETION_UNIT)
        legal_actions.insert(0, DEPENDENCY_COMPLETION_UNIT)

    forbidden: list[str] = []
    if residual_request.get("requires_illegal_side_effect_avoidance"):
        forbidden.append(MINEFIELD_SIDE_EFFECT_UNIT)

    return ToolSandboxResidualCertificate(
        visible_prefix={
            "user_turns_visible": bool(visible_prefix.get("user_turns_visible")),
            "assistant_turns_visible": bool(visible_prefix.get("assistant_turns_visible")),
            "visible_tool_result_count": _nonnegative_int(
                visible_prefix.get("visible_tool_result_count")
            ),
        },
        evidence=evidence,
        commitments=tuple(commitments),
        residual_obligations=tuple(residual_obligations),
        legal_residual_actions=tuple(legal_actions),
        authority={
            "matched_strong_call_budget": _nonnegative_int(
                residual_request.get("matched_strong_call_budget")
            ),
            "max_residual_tool_calls": _nonnegative_int(
                residual_request.get("max_residual_tool_calls")
            ),
            "uses_visible_prefix_only": certificate_digest.get("uses_visible_prefix_only")
            is True,
            "uses_scorer_side_labels": certificate_digest.get("uses_scorer_side_labels")
            is True,
            "hidden_snapshot_used": observed_state.get("hidden_snapshot_used") is True,
        },
        forbidden_effects=tuple(forbidden),
        provider_calls=0,
    )


def certificate_leakage_markers(certificate: Mapping[str, Any]) -> list[str]:
    """Return forbidden marker names present in a public certificate payload."""

    text = repr(certificate)
    markers = (
        "task_id",
        "gold",
        "raw_prompt",
        "raw_response",
        "scenario_name",
        "milestones",
        "minefields",
        "expected_tool_arguments",
        "target_dataframe",
        f"{DATA_ROOT}",
        "OPENAI_API_KEY",
        "sk-",
    )
    return sorted(marker for marker in markers if marker in text)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return value
    return ()


__all__ = [
    "ToolSandboxResidualCertificate",
    "certificate_from_runtime_digest",
    "certificate_leakage_markers",
]

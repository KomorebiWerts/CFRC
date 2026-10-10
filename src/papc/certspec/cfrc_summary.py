"""Public-safe summaries for Commitment-Frontier Residual Completion.

This module is a paper-facing layer over existing certspec objects. It does not
execute actions, change projection semantics, or expose raw benchmark/runtime
content. It maps the implementation vocabulary (state-write contracts, proposed
actions, and action projections) to the CFRC notation used in docs:
``z_f -> H(a) -> Gamma_z -> Pi_K -> residual suffix / abstain``.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from papc.certspec.shared_abstractions import find_forbidden_runtime_references
from papc.certspec.state_write_contract import (
    ActionProjection,
    ProposedTargetAction,
    StateWriteContract,
)

CFRCExecutionFamily = Literal[
    "strict_residual",
    "scoped_continuation",
    "loop_refinement",
    "authority_extension",
    "abstain",
]
# Historical exported name.  The values now denote execution-family accounting,
# not a paper-facing routing method.
CFRCRouteFamily = CFRCExecutionFamily
CFRCAuthorityLevel = Literal[
    "accept",
    "read_only_completion",
    "residual_patch",
    "dependent_residual_bundle",
    "scoped_continuation",
    "full_agent_completion",
    "abstain",
]

CFRC_DECISION_SUMMARY_SCHEMA = "papc.cfrc_decision_summary.v2"
CFRC_ROUTE_DISTRIBUTION_SCHEMA = "papc.cfrc_execution_family_distribution.v1"

_PRIVATE_PATH_RE = re.compile(
    r"/(?:home|mnt)/(?:runzhi|user-data|transcripts|skills)(?:/|$)"
)
_SECRET_MARKER_RE = re.compile(
    r"(?:api[_-]?key|authorization|bearer\s+[a-z0-9._-]+|secret|token)",
    flags=re.IGNORECASE,
)
_FORBIDDEN_SUMMARY_KEYS = frozenset(
    {
        "arguments",
        "compiled_solution",
        "db_final_state",
        "direct_outcome",
        "expected_answer",
        "final_state",
        "fixed_api_args",
        "gold",
        "gold_action_sequence",
        "gold_api_trace",
        "ground_truth",
        "hidden",
        "private_data",
        "prompt",
        "provider_payload",
        "raw_prompt",
        "raw_response",
        "raw_trajectory",
        "reference_trajectory",
        "scorer_text",
        "solution",
        "state_diff",
        "target_answer",
        "target_state_summary",
        "task_id",
        "tool_calls",
    }
)


def summarize_cfrc_decision(
    *,
    contract: StateWriteContract,
    action: ProposedTargetAction | None = None,
    projection: ActionProjection | None = None,
    route_family: str | None = None,
    authority_level: str | None = None,
    provider_calls: int | None = None,
    b_call_total: int | None = None,
    b_call_spent: int | None = None,
    surface: str | None = None,
) -> dict[str, Any]:
    """Return a public-safe CFRC summary over existing certspec objects.

    The summary intentionally reports counts and canonical execution labels
    rather than raw unit names, task identifiers, prompts, tool arguments, or
    provider payloads. ``route_family`` is retained as an input parameter for
    historical callers; public output uses ``execution_family``.
    """

    execution_family = canonical_execution_family(
        route_family,
        projection=projection,
    )
    canonical_authority = canonical_authority_level(
        authority_level,
        route_family=execution_family,
        action=action,
        projection=projection,
    )
    payload: dict[str, Any] = {
        "schema": CFRC_DECISION_SUMMARY_SCHEMA,
        "surface": str(surface or contract.surface),
        "execution_family": execution_family,
        "authority_level": canonical_authority,
        "provider_calls": _nonnegative_int(provider_calls),
        "b_call_total": _optional_nonnegative_int(b_call_total),
        "b_call_spent": _optional_nonnegative_int(b_call_spent),
        "z_summary": _summarize_z(contract),
        "h_summary": _summarize_h(action),
        "pik_summary": _summarize_pik(projection),
    }
    leaks = audit_cfrc_public_summary(payload)
    if leaks:
        raise ValueError(f"CFRC summary contains forbidden public fields: {leaks}")
    return payload


def summarize_execution_family_distribution(
    rows: Iterable[Mapping[str, Any] | str],
) -> dict[str, Any]:
    """Aggregate execution families for public reporting.

    Rows may be dictionaries from reducer outputs or direct label strings.
    Unknown labels are conservatively counted as ``abstain``.
    """

    counts: Counter[str] = Counter({family: 0 for family in _EXECUTION_FAMILY_ORDER})
    for row in rows:
        counts[_execution_family_from_row(row)] += 1
    total = sum(counts.values())
    ordered_counts = {family: counts[family] for family in _EXECUTION_FAMILY_ORDER}
    return {
        "schema": CFRC_ROUTE_DISTRIBUTION_SCHEMA,
        "execution_family_counts": ordered_counts,
        "total": total,
    }


def summarize_route_distribution(
    rows: Iterable[Mapping[str, Any] | str],
) -> dict[str, Any]:
    """Compatibility wrapper for execution-family accounting.

    The historical name is retained because several reducers call it. New method
    text should call this an execution-family distribution, not route selection.
    """

    return summarize_execution_family_distribution(rows)


def canonical_route_family(
    route: str | None,
    *,
    projection: ActionProjection | None = None,
) -> CFRCRouteFamily:
    """Compatibility wrapper for ``canonical_execution_family``."""

    return canonical_execution_family(route, projection=projection)


def canonical_execution_family(
    route: str | None,
    *,
    projection: ActionProjection | None = None,
) -> CFRCExecutionFamily:
    """Normalize implementation labels to CFRC execution families.

    Historical ``fallback`` / ``full_fallback`` labels are accepted as aliases,
    but canonical public output uses ``authority_extension``.
    """

    text = str(route or "").strip().lower().replace("-", "_")
    if text in {
        "strict",
        "strict_residual",
        "cfrc_strict",
        "execute_projected_suffix",
        "projected_suffix",
    }:
        return "strict_residual"
    if text in {
        "adaptive",
        "scoped_continuation",
        "cfrc_adaptive",
        "authority_adaptive",
    }:
        return "scoped_continuation"
    if text in {
        "loop",
        "loop_refinement",
        "cfrc_loop",
        "verifier_loop",
        "residual_loop",
    }:
        return "loop_refinement"
    if text in {
        "fallback",
        "full_fallback",
        "direct_strong",
        "strong_full_agent",
        "direct_strong_fallback",
        "direct_strong_full_agent",
        "authority_extension",
        "full_agent",
    }:
        return "authority_extension"
    if text in {"abstain", "stop", "blocked", "project_abstain"}:
        return "abstain"
    if projection is not None:
        return "strict_residual" if projection.makes_progress else "abstain"
    return "abstain"


def canonical_authority_level(
    label: str | None = None,
    *,
    route_family: str | None = None,
    action: ProposedTargetAction | None = None,
    projection: ActionProjection | None = None,
) -> CFRCAuthorityLevel:
    """Normalize route/action/projection hints to the CFRC authority family."""

    text = str(label or route_family or "").strip().lower().replace("-", "_")
    if text in {"accept", "a0", "none"}:
        return "accept"
    if text in {"read_only", "read_only_completion", "deterministic_evidence", "a1"}:
        return "read_only_completion"
    if text in {
        "residual_patch",
        "bounded_repair",
        "repair",
        "strict",
        "strict_residual",
        "a2",
    }:
        return "residual_patch"
    if text in {
        "dependent_residual_bundle",
        "dependency_bundle",
        "loop",
        "loop_refinement",
        "cfrc_loop",
        "a3",
    }:
        return "dependent_residual_bundle"
    if text in {"scoped_continuation", "adaptive", "cfrc_adaptive", "a4"}:
        return "scoped_continuation"
    if text in {
        "fallback",
        "full_fallback",
        "direct_strong",
        "strong_full_agent",
        "direct_strong_fallback",
        "direct_strong_full_agent",
        "authority_extension",
        "full_agent",
        "full_agent_completion",
        "a5",
    }:
        return "full_agent_completion"
    if text in {"abstain", "stop", "blocked", "project_abstain"}:
        return "abstain"
    if projection is not None and not projection.makes_progress:
        return "abstain"
    if action is not None and action.has_closure_structure():
        return "dependent_residual_bundle"
    if action is not None and action.write_footprint:
        return "residual_patch"
    return "abstain"


def audit_cfrc_public_summary(payload: Mapping[str, Any]) -> list[str]:
    """Return public-safety findings for a CFRC summary payload."""

    leaks = list(
        find_forbidden_runtime_references(
            payload,
            forbidden_keys=_FORBIDDEN_SUMMARY_KEYS,
            forbidden_string_markers=tuple(sorted(_FORBIDDEN_SUMMARY_KEYS)),
        )
    )
    rendered = json.dumps(payload, sort_keys=True, default=str)
    if _PRIVATE_PATH_RE.search(rendered):
        leaks.append("$.private_path")
    if _SECRET_MARKER_RE.search(rendered):
        leaks.append("$.secret_marker")
    return sorted(set(leaks))


_EXECUTION_FAMILY_ORDER: tuple[CFRCExecutionFamily, ...] = (
    "strict_residual",
    "scoped_continuation",
    "loop_refinement",
    "authority_extension",
    "abstain",
)


def _summarize_z(contract: StateWriteContract) -> dict[str, int]:
    committed_units = (
        frozenset(unit for obligation in contract.closed for unit in obligation.req())
        | contract.forbidden_writes
    )
    residual_units = frozenset(
        unit for obligation in contract.residual for unit in obligation.req()
    )
    return {
        "k_commitment_unit_count": len(committed_units),
        "closed_obligation_count": len(contract.closed),
        "residual_obligation_count": len(contract.residual),
        "residual_unit_count": len(residual_units),
        "observed_read_unit_count": len(contract.observed_reads),
        "touched_unit_count": len(contract.touched_units),
    }


def _summarize_h(action: ProposedTargetAction | None) -> dict[str, int | bool]:
    if action is None:
        return {
            "proposed_effect_unit_count": 0,
            "effect_group_count": 0,
            "dependency_edge_count": 0,
            "has_dependency_structure": False,
        }
    return {
        "proposed_effect_unit_count": len(action.effects()),
        "effect_group_count": len(action.effect_groups),
        "dependency_edge_count": len(action.dependencies),
        "has_dependency_structure": action.has_closure_structure(),
    }


def _summarize_pik(
    projection: ActionProjection | None,
) -> dict[str, int | float | bool | str]:
    if projection is None:
        return {
            "projection_decision": "unavailable",
            "projected_unit_count": 0,
            "trimmed_unit_count": 0,
            "progress_units": 0,
            "discharged_obligation_count": 0,
            "lambda_collision": 0.0,
            "makes_progress": False,
        }
    return {
        "projection_decision": projection.decision,
        "projected_unit_count": len(projection.projected_footprint),
        "trimmed_unit_count": len(projection.trimmed_footprint),
        "progress_units": projection.progress_units,
        "discharged_obligation_count": projection.discharged_obligation_count,
        "lambda_collision": projection.lambda_collision,
        "makes_progress": projection.makes_progress,
    }


def _execution_family_from_row(row: Mapping[str, Any] | str) -> CFRCExecutionFamily:
    if isinstance(row, str):
        return canonical_execution_family(row)
    for key in (
        "execution_family",
        "route_family",
        "cfrc_route_family",
        "route",
        "decision_route",
        "decision",
        "arm",
    ):
        value = row.get(key)
        if value is not None:
            return canonical_execution_family(str(value))
    if row.get("fallback_used") is True:
        return "authority_extension"
    if row.get("abstained") is True:
        return "abstain"
    return "abstain"


def _nonnegative_int(value: int | None) -> int:
    if value is None:
        return 0
    return max(0, int(value))


def _optional_nonnegative_int(value: int | None) -> int | None:
    if value is None:
        return None
    return max(0, int(value))


__all__ = [
    "CFRC_DECISION_SUMMARY_SCHEMA",
    "CFRC_ROUTE_DISTRIBUTION_SCHEMA",
    "CFRCAuthorityLevel",
    "CFRCExecutionFamily",
    "CFRCRouteFamily",
    "audit_cfrc_public_summary",
    "canonical_authority_level",
    "canonical_execution_family",
    "canonical_route_family",
    "summarize_execution_family_distribution",
    "summarize_route_distribution",
    "summarize_cfrc_decision",
]

"""Structured certificate primitives for PAPC v2.

This module is intentionally benchmark-independent.  It turns existing
shared-field certificates into a graph-like certificate with abstract targets,
actions, constraints, residuals, and an ordered repair plan.  Concrete task ids,
entity ids, fixed tool arguments, gold outcomes, and direct outcomes must remain
outside the model-visible surface.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from papc.certspec.shared_abstractions import (
    SHARED_CERTIFICATE_FIELDS,
    find_forbidden_runtime_references,
)

STRUCTURED_CERTIFICATE_SCHEMA = "papc_structured_certificate.v1"
STRUCTURED_REPAIR_DELTA_SCHEMA = "papc_structured_repair_delta.v1"
STRUCTURED_REPAIR_PACKET_SCHEMA = "papc_structured_repair_verifier_packet.v1"
STRUCTURED_REPAIR_POLICY = "bounded_delta_ordered_repair_v1"
STRUCTURED_ACTION_OBLIGATION_POLICY = "abstract_action_obligation_v1"

DEFAULT_TARGET_TYPE = "workflow_target_set"

_LEGACY_REASON_TO_RESIDUAL_REGISTRY: dict[tuple[str, str], dict[str, str]] = {}
_RESIDUAL_VOCABULARY_REGISTRY: dict[tuple[str, str], tuple[str, ...]] = {}
_TARGET_TYPE_REGISTRY: dict[tuple[str, str], str] = {}


def register_legacy_reason_to_residual(
    benchmark: str,
    domain: str,
    mapping: Mapping[str, str],
    *,
    residual_vocabulary: Sequence[str] | None = None,
) -> None:
    """Register an adapter-supplied legacy-reason → structured-residual mapping.

    Core stays benchmark-agnostic: the only generic mapping it provides is the
    status-driven path in ``residuals_from_legacy_certificate``. Adapters that
    carry historical reason vocabularies (for example tau2 retail) register
    them here, with a local residual vocabulary when needed, so
    ``structured_certificate_from_legacy`` can look them up by (benchmark,
    domain) without any domain switch in core.

    The registry is a process-global ``dict`` populated at adapter import
    time. It is not thread-safe; callers that mutate it concurrently must
    synchronize externally. The expected pattern is "register at import,
    read during request handling".
    """

    _validate_legacy_reason_mapping(mapping, residual_vocabulary=residual_vocabulary)
    key = (str(benchmark or "unknown"), str(domain or "unknown"))
    _LEGACY_REASON_TO_RESIDUAL_REGISTRY[key] = dict(mapping)
    if residual_vocabulary is not None:
        _RESIDUAL_VOCABULARY_REGISTRY[key] = _residual_vocabulary(residual_vocabulary)


def get_legacy_reason_to_residual(
    benchmark: str | None,
    domain: str | None,
) -> dict[str, str]:
    """Return a copy of the registered mapping for (benchmark, domain), or empty."""

    key = (str(benchmark or "unknown"), str(domain or "unknown"))
    return dict(_LEGACY_REASON_TO_RESIDUAL_REGISTRY.get(key, {}))


def get_residual_vocabulary(
    benchmark: str | None,
    domain: str | None,
) -> tuple[str, ...] | None:
    """Return the adapter-registered residual vocabulary for (benchmark, domain)."""

    key = (str(benchmark or "unknown"), str(domain or "unknown"))
    return _RESIDUAL_VOCABULARY_REGISTRY.get(key)


def register_target_type(benchmark: str, domain: str, target_type: str) -> None:
    """Register an adapter-supplied workflow target_type label.

    The label is purely metadata that flows into structured certificate
    payloads. Core defaults to ``DEFAULT_TARGET_TYPE`` when no adapter has
    registered a more specific label.
    """

    key = (str(benchmark or "unknown"), str(domain or "unknown"))
    _TARGET_TYPE_REGISTRY[key] = str(target_type or DEFAULT_TARGET_TYPE)


def get_target_type(benchmark: str | None, domain: str | None) -> str:
    """Return the registered target_type for (benchmark, domain) or default."""

    key = (str(benchmark or "unknown"), str(domain or "unknown"))
    return _TARGET_TYPE_REGISTRY.get(key, DEFAULT_TARGET_TYPE)


STRUCTURED_CERTIFICATE_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "benchmark",
        "domain",
        "source",
        "targets",
        "actions",
        "constraints",
        "residuals",
        "repair_plan",
        "legacy_projection",
        "leakage_audit",
        "identifier_policy",
    }
)

GENERIC_RESIDUAL_TYPES: tuple[str, ...] = (
    "evidence_sweep",
    "target_set_binding",
    "argument_binding",
    "precondition_binding",
    "source_read_binding",
    "selection_target_binding",
    "write_argument_binding",
    "argument_preference_binding",
    "terminal_completion",
    "postcondition_risk",
    "policy_boundary",
    "unknown_structured_residual",
)

REPAIR_STAGE_ORDER: tuple[str, ...] = (
    "evidence",
    "user_choice",
    "argument_binding",
    "terminal_write",
    "postcondition",
)

RESIDUAL_TO_REPAIR_STAGE: Mapping[str, str] = {
    "evidence_sweep": "evidence",
    "target_set_binding": "evidence",
    "precondition_binding": "evidence",
    "source_read_binding": "evidence",
    "selection_target_binding": "evidence",
    "argument_binding": "argument_binding",
    "write_argument_binding": "argument_binding",
    "argument_preference_binding": "user_choice",
    "terminal_completion": "terminal_write",
    "postcondition_risk": "postcondition",
    "policy_boundary": "user_choice",
    "unknown_structured_residual": "evidence",
}

RESIDUAL_TO_CONSTRAINT: Mapping[str, str] = {
    "evidence_sweep": "requires_visible_evidence_sweep",
    "target_set_binding": "requires_complete_target_set",
    "precondition_binding": "requires_precondition_binding",
    "source_read_binding": "requires_source_read_binding",
    "selection_target_binding": "requires_selection_target_binding",
    "argument_binding": "requires_complete_argument_binding",
    "write_argument_binding": "requires_write_argument_binding",
    "argument_preference_binding": "requires_user_preference_binding",
    "terminal_completion": "requires_terminal_completion",
    "postcondition_risk": "requires_postcondition_verification",
    "policy_boundary": "requires_policy_boundary_resolution",
    "unknown_structured_residual": "requires_adapter_mapping_review",
}

RESIDUAL_TO_OPERATION_FAMILY: Mapping[str, str] = {
    "evidence_sweep": "evidence_sweep",
    "target_set_binding": "evidence_sweep",
    "precondition_binding": "precondition_binding",
    "source_read_binding": "source_read_binding",
    "selection_target_binding": "selection_target_binding",
    "argument_binding": "argument_binding",
    "write_argument_binding": "write_argument_binding",
    "argument_preference_binding": "user_choice",
    "terminal_completion": "terminal_write",
    "postcondition_risk": "postcondition_check",
    "policy_boundary": "user_choice",
    "unknown_structured_residual": "adapter_mapping_review",
}

RESIDUAL_TO_ARGUMENT_SOURCE: Mapping[str, str] = {
    "evidence_sweep": "visible_evidence",
    "target_set_binding": "visible_target_set",
    "precondition_binding": "visible_precondition",
    "source_read_binding": "visible_source_read",
    "selection_target_binding": "visible_selection_target",
    "argument_binding": "visible_bound_argument",
    "write_argument_binding": "visible_write_argument",
    "argument_preference_binding": "latest_user_choice_or_ask_user",
    "terminal_completion": "visible_bound_argument",
    "postcondition_risk": "postcondition_state",
    "policy_boundary": "policy_visible_evidence",
    "unknown_structured_residual": "adapter_mapping_review",
}

RESIDUAL_TO_TARGET_SELECTION_RULE: Mapping[str, str] = {
    "evidence_sweep": "sweep_visible_targets_before_commit",
    "target_set_binding": "all_visible_targets",
    "precondition_binding": "targets_with_unmet_preconditions",
    "source_read_binding": "targets_with_unconfirmed_source_reads",
    "selection_target_binding": "targets_with_unconfirmed_selection",
    "argument_binding": "bound_targets_with_incomplete_arguments",
    "write_argument_binding": "write_targets_with_unconfirmed_arguments",
    "argument_preference_binding": "target_requiring_user_choice",
    "terminal_completion": "missing_terminal_write_targets",
    "postcondition_risk": "candidate_write_surface",
    "policy_boundary": "target_requiring_policy_resolution",
    "unknown_structured_residual": "adapter_mapping_review",
}

RESIDUAL_TO_COMPLETION_CHECK: Mapping[str, str] = {
    "evidence_sweep": "visible_evidence_swept",
    "target_set_binding": "complete_target_set_bound",
    "precondition_binding": "preconditions_bound_before_commit",
    "source_read_binding": "source_reads_confirmed",
    "selection_target_binding": "selection_targets_bound",
    "argument_binding": "arguments_bound_before_write",
    "write_argument_binding": "write_arguments_bound_before_commit",
    "argument_preference_binding": "user_choice_bound_or_safe_question_asked",
    "terminal_completion": "terminal_write_completed",
    "postcondition_risk": "postcondition_verified",
    "policy_boundary": "policy_boundary_resolved",
    "unknown_structured_residual": "adapter_mapping_reviewed",
}


def legacy_shared_projection(certificate: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the historical shared-field projection."""

    return {
        field: _json_safe(certificate.get(field, {}))
        for field in SHARED_CERTIFICATE_FIELDS
        if field in certificate
    }


def structured_certificate_from_legacy(
    certificate: Mapping[str, Any],
    *,
    benchmark: str = "unknown",
    domain: str = "unknown",
    source: str = "legacy_shared_fields",
    reason_to_residual: Mapping[str, str] | None = None,
    residual_vocabulary: Sequence[str] | None = None,
    residual_to_repair_stage: Mapping[str, str] | None = None,
    residual_to_constraint: Mapping[str, str] | None = None,
    residual_evidence_state: Callable[[str, Mapping[str, Any]], str] | None = None,
    residual_severity_overrides: Mapping[str, str] | None = None,
    target_type: str | None = None,
) -> dict[str, Any]:
    """Build a structured certificate from an existing shared-field certificate.

    ``reason_to_residual`` and ``target_type`` may be passed explicitly by an
    adapter. When omitted, the registries populated by
    ``register_legacy_reason_to_residual`` and ``register_target_type`` are
    consulted using ``(benchmark, domain)``. Core itself never holds a
    domain-specific reason vocabulary or target_type label.
    """

    legacy = legacy_shared_projection(certificate)
    target = _mapping(legacy.get("target_binding"))
    argument = _mapping(legacy.get("argument_binding"))
    precondition = _mapping(legacy.get("precondition_state"))
    commit = _mapping(legacy.get("commit_contract"))
    postcondition = _mapping(legacy.get("postcondition"))
    terminal_plan = _mapping(legacy.get("terminal_completion_plan"))
    if reason_to_residual is None:
        reason_to_residual = get_legacy_reason_to_residual(benchmark, domain)
    if residual_vocabulary is None:
        residual_vocabulary = get_residual_vocabulary(benchmark, domain)
    residuals = residuals_from_legacy_certificate(
        legacy,
        reason_to_residual=reason_to_residual,
        residual_vocabulary=residual_vocabulary,
        residual_evidence_state=residual_evidence_state,
        residual_severity_overrides=residual_severity_overrides,
    )
    residual_types = [str(row["residual_type"]) for row in residuals]
    constraints = constraints_from_residual_types(
        residual_types,
        residual_vocabulary=residual_vocabulary,
        residual_to_constraint=residual_to_constraint,
    )
    actions = actions_from_legacy_certificate(
        residual_types,
        precondition=precondition,
        terminal_plan=terminal_plan,
        postcondition=postcondition,
        residual_vocabulary=residual_vocabulary,
        residual_to_repair_stage=residual_to_repair_stage,
    )
    repair_plan = repair_plan_from_residual_types(
        residual_types,
        residual_vocabulary=residual_vocabulary,
        residual_to_repair_stage=residual_to_repair_stage,
    )
    resolved_target_type = (
        str(target_type) if target_type else get_target_type(benchmark, domain)
    )
    targets = [
        {
            "target_ref": "target_set:primary",
            "target_type": resolved_target_type,
            "binding_state": _first_nonempty(
                target.get("target_binding_status"),
                target.get("target_sweep_status"),
                target.get("identity_binding"),
                "unknown",
            ),
            "scope": _target_scope(target, commit),
            "count_state": {
                "candidate_count": _int_value(target.get("target_count")),
                "observed_count": _int_value(target.get("observed_count")),
                "completed_count": _int_value(
                    target.get("completed_count") or commit.get("completed_write_count")
                ),
                "expected_write_count": _int_value(commit.get("expected_write_count")),
            },
            "attributes": _compact_mapping(
                {
                    "identity_binding": target.get("identity_binding"),
                    "account_binding": target.get("account_binding"),
                    "target_conflict": target.get("target_conflict"),
                    "argument_binding_status": argument.get("argument_binding_status"),
                    "tool_evidence_status": precondition.get("tool_evidence_status"),
                    "terminal_completion_status": postcondition.get(
                        "terminal_completion_status"
                    ),
                }
            ),
        }
    ]
    structured = {
        "schema": STRUCTURED_CERTIFICATE_SCHEMA,
        "benchmark": str(benchmark or "unknown"),
        "domain": str(domain or "unknown"),
        "source": str(source or "legacy_shared_fields"),
        "targets": targets,
        "actions": actions,
        "constraints": constraints,
        "residuals": residuals,
        "repair_plan": repair_plan,
        "legacy_projection": legacy,
        "identifier_policy": (
            "model-visible structured certificates expose abstract target/action "
            "aliases, counts, states, constraints, residuals, and repair stages; "
            "task ids, gold/direct outcomes, concrete entity ids, fixed tool "
            "arguments, and answer sketches are omitted"
        ),
    }
    leakage_refs = structured_certificate_leakage_refs(structured)
    structured["leakage_audit"] = {
        "audited": True,
        "leaked": bool(leakage_refs),
        "forbidden_paths": leakage_refs,
        "policy": "papc_structured_certificate_no_runtime_leakage_v1",
    }
    return structured


def residuals_from_legacy_certificate(
    certificate: Mapping[str, Any],
    *,
    reason_to_residual: Mapping[str, str] | None = None,
    residual_vocabulary: Sequence[str] | None = None,
    residual_evidence_state: Callable[[str, Mapping[str, Any]], str] | None = None,
    residual_severity_overrides: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Map legacy risk/status fields to generic structured residuals.

    ``reason_to_residual`` is an adapter-supplied vocabulary mapping legacy
    risk reasons to generic residual types. When omitted, only the
    status-driven, vocabulary-free residuals are produced.
    """

    reason_map: Mapping[str, str] = reason_to_residual or {}
    _validate_legacy_reason_mapping(
        reason_map,
        residual_vocabulary=residual_vocabulary,
    )
    risk = _mapping(certificate.get("risk"))
    risk_reasons = _string_list(risk.get("reasons"))
    residual_types: list[str] = []
    source_reasons: dict[str, list[str]] = {}
    for reason in risk_reasons:
        residual_type = reason_map.get(
            reason,
            _status_reason_to_residual(reason),
        )
        residual_types.append(residual_type)
        source_reasons.setdefault(residual_type, []).append(reason)

    target = _mapping(certificate.get("target_binding"))
    argument = _mapping(certificate.get("argument_binding"))
    precondition = _mapping(certificate.get("precondition_state"))
    postcondition = _mapping(certificate.get("postcondition"))
    terminal_plan = _mapping(certificate.get("terminal_completion_plan"))
    _add_status_residual(
        residual_types,
        source_reasons,
        residual_type="target_set_binding",
        source="target_binding_status",
        value=target.get("target_binding_status") or target.get("target_sweep_status"),
        bad_values={"missing", "partial", "conflicting", "incomplete"},
    )
    _add_status_residual(
        residual_types,
        source_reasons,
        residual_type="argument_binding",
        source="argument_binding_status",
        value=argument.get("argument_binding_status"),
        bad_values={"missing", "partial", "conflicting", "incomplete", "unknown"},
    )
    _add_status_residual(
        residual_types,
        source_reasons,
        residual_type="evidence_sweep",
        source="tool_evidence_status",
        value=precondition.get("tool_evidence_status"),
        bad_values={
            "missing_auth",
            "tool_error",
            "insufficient",
            "incomplete",
            "partial",
        },
    )
    _add_status_residual(
        residual_types,
        source_reasons,
        residual_type="terminal_completion",
        source="terminal_completion_status",
        value=postcondition.get("terminal_completion_status")
        or terminal_plan.get("terminal_completion_status"),
        bad_values={"missing", "partial", "conflicting"},
    )
    if not residual_types:
        return []

    severity_by_type = _severity_by_type(
        risk.get("level"),
        residual_severity_overrides=residual_severity_overrides,
    )
    ordered = [
        residual_type
        for residual_type in _residual_vocabulary(residual_vocabulary)
        if residual_type in set(residual_types)
    ]
    evidence_state = residual_evidence_state or _residual_evidence_state
    return [
        {
            "residual_ref": f"residual:{index}:{residual_type}",
            "residual_type": residual_type,
            "severity": severity_by_type.get(residual_type, "medium"),
            "target_refs": ["target_set:primary"],
            "action_refs": [],
            "constraint_refs": [f"constraint:{residual_type}"],
            "evidence_state": evidence_state(residual_type, certificate),
            "source_reasons": sorted(set(source_reasons.get(residual_type, []))),
        }
        for index, residual_type in enumerate(ordered, start=1)
    ]


def _validate_legacy_reason_mapping(
    mapping: Mapping[str, str],
    *,
    residual_vocabulary: Sequence[str] | None = None,
) -> None:
    allowed = set(_residual_vocabulary(residual_vocabulary))
    unknown = sorted(
        {
            str(residual_type)
            for residual_type in mapping.values()
            if str(residual_type) not in allowed
        }
    )
    if unknown:
        raise ValueError(
            f"legacy reason mapping uses unknown generic residuals: {unknown}"
        )


def constraints_from_residual_types(
    residual_types: Sequence[str],
    *,
    residual_vocabulary: Sequence[str] | None = None,
    residual_to_constraint: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = []
    constraint_map = _merge_residual_mapping(
        RESIDUAL_TO_CONSTRAINT,
        residual_to_constraint,
    )
    for residual_type in _ordered_unique_residual_types(
        residual_types,
        residual_vocabulary=residual_vocabulary,
    ):
        constraints.append(
            {
                "constraint_ref": f"constraint:{residual_type}",
                "constraint_type": constraint_map.get(
                    residual_type,
                    "requires_adapter_mapping_review",
                ),
                "scope": "abstract_workflow",
                "status": "open",
                "target_refs": ["target_set:primary"],
                "action_refs": [],
                "residual_type": residual_type,
            }
        )
    return constraints


def actions_from_legacy_certificate(
    residual_types: Sequence[str],
    *,
    precondition: Mapping[str, Any],
    terminal_plan: Mapping[str, Any],
    postcondition: Mapping[str, Any],
    residual_vocabulary: Sequence[str] | None = None,
    residual_to_repair_stage: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    stage_map = _merge_residual_mapping(
        RESIDUAL_TO_REPAIR_STAGE,
        residual_to_repair_stage,
    )
    stages = {
        stage_map.get(residual_type, "evidence") for residual_type in residual_types
    }
    if terminal_plan.get("requires_more_tools"):
        stages.add("evidence")
    if terminal_plan.get("requires_user_input"):
        stages.add("user_choice")
    if terminal_plan.get("requires_strong_model"):
        stages.add("terminal_write")
    if postcondition.get("terminal_completion_status") in {"missing", "partial"}:
        stages.add("terminal_write")
        stages.add("postcondition")
    if precondition.get("tool_evidence_status") in {"missing_auth", "incomplete"}:
        stages.add("evidence")

    return [
        {
            "action_ref": f"action:{stage}",
            "action_type": _action_type_for_stage(stage),
            "stage": stage,
            "status": "pending",
            "target_refs": ["target_set:primary"],
            "constraint_refs": [
                f"constraint:{residual_type}"
                for residual_type in _ordered_unique_residual_types(
                    residual_types,
                    residual_vocabulary=residual_vocabulary,
                )
                if stage_map.get(residual_type, "evidence") == stage
            ],
        }
        for stage in REPAIR_STAGE_ORDER
        if stage in stages
    ]


def repair_plan_from_residual_types(
    residual_types: Sequence[str],
    *,
    residual_vocabulary: Sequence[str] | None = None,
    residual_to_repair_stage: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    stage_map = _merge_residual_mapping(
        RESIDUAL_TO_REPAIR_STAGE,
        residual_to_repair_stage,
    )
    by_stage: dict[str, list[str]] = {stage: [] for stage in REPAIR_STAGE_ORDER}
    ordered_residual_types = _ordered_unique_residual_types(
        residual_types,
        residual_vocabulary=residual_vocabulary,
    )
    for residual_type in ordered_residual_types:
        stage = stage_map.get(residual_type, "evidence")
        by_stage.setdefault(stage, []).append(residual_type)
        if stage == "terminal_write":
            by_stage.setdefault("postcondition", []).append(residual_type)
    steps: list[dict[str, Any]] = []
    for stage in REPAIR_STAGE_ORDER:
        stage_residuals = by_stage.get(stage) or []
        if not stage_residuals:
            continue
        steps.append(
            {
                "step_ref": f"repair_step:{len(steps) + 1}:{stage}",
                "stage": stage,
                "action_ref": f"action:{stage}",
                "residual_refs": [
                    f"residual:{index}:{residual_type}"
                    for index, residual_type in enumerate(
                        ordered_residual_types,
                        start=1,
                    )
                    if residual_type in stage_residuals
                ],
                "constraint_refs": [
                    f"constraint:{residual_type}" for residual_type in stage_residuals
                ],
                "required_side": _required_side_for_stage(stage),
                "status": "open",
            }
        )
    return steps


def model_visible_structured_certificate(
    certificate: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the safe model-visible structured certificate surface."""

    visible = {
        key: _json_safe(certificate.get(key))
        for key in (
            "schema",
            "benchmark",
            "domain",
            "source",
            "targets",
            "actions",
            "constraints",
            "residuals",
            "repair_plan",
            "identifier_policy",
        )
        if key in certificate
    }
    return visible


def structured_certificate_leakage_refs(certificate: Mapping[str, Any]) -> list[str]:
    """Check the model-visible structured surface for forbidden references."""

    return find_forbidden_runtime_references(
        model_visible_structured_certificate(certificate)
    )


def validate_structured_certificate(certificate: Mapping[str, Any]) -> None:
    """Raise ValueError if the structured certificate surface is malformed."""

    extra = sorted(set(certificate) - STRUCTURED_CERTIFICATE_TOP_LEVEL_FIELDS)
    if extra:
        raise ValueError(f"structured certificate contains unknown fields: {extra}")
    if certificate.get("schema") != STRUCTURED_CERTIFICATE_SCHEMA:
        raise ValueError("structured certificate has unsupported schema")
    leakage_refs = structured_certificate_leakage_refs(certificate)
    if leakage_refs:
        raise ValueError(f"structured certificate leaks forbidden refs: {leakage_refs}")


def summarize_structured_certificate(certificate: Mapping[str, Any]) -> dict[str, Any]:
    residuals = [
        str(_mapping(item).get("residual_type"))
        for item in _sequence(certificate.get("residuals"))
        if _mapping(item).get("residual_type")
    ]
    stages = [
        str(_mapping(item).get("stage"))
        for item in _sequence(certificate.get("repair_plan"))
        if _mapping(item).get("stage")
    ]
    constraints = [
        str(_mapping(item).get("constraint_type"))
        for item in _sequence(certificate.get("constraints"))
        if _mapping(item).get("constraint_type")
    ]
    return {
        "schema": certificate.get("schema"),
        "benchmark": certificate.get("benchmark"),
        "domain": certificate.get("domain"),
        "source": certificate.get("source"),
        "target_count": len(_sequence(certificate.get("targets"))),
        "action_count": len(_sequence(certificate.get("actions"))),
        "constraint_count": len(_sequence(certificate.get("constraints"))),
        "residual_count": len(residuals),
        "residual_type_counts": dict(sorted(Counter(residuals).items())),
        "repair_stages": stages,
        "constraint_type_counts": dict(sorted(Counter(constraints).items())),
        "leakage_refs": structured_certificate_leakage_refs(certificate),
    }


def structured_repair_delta(
    certificate: Mapping[str, Any],
    *,
    residual_vocabulary: Sequence[str] | None = None,
    residual_to_operation_family: Mapping[str, str] | None = None,
    residual_to_argument_source: Mapping[str, str] | None = None,
    residual_to_target_selection_rule: Mapping[str, str] | None = None,
    residual_to_completion_check: Mapping[str, str] | None = None,
    obligation_count_state: Callable[[str, Mapping[str, Any]], dict[str, int]]
    | None = None,
    obligation_ordering: Callable[[str, Sequence[str]], list[str]] | None = None,
) -> dict[str, Any]:
    """Return the bounded ordered delta a verifier may inspect or repair."""

    residuals_by_ref = {
        str(row.get("residual_ref") or ""): _mapping(row)
        for row in _sequence(certificate.get("residuals"))
        if _mapping(row).get("residual_ref")
    }
    constraints_by_ref = {
        str(row.get("constraint_ref") or ""): _mapping(row)
        for row in _sequence(certificate.get("constraints"))
        if _mapping(row).get("constraint_ref")
    }
    actions_by_ref = {
        str(row.get("action_ref") or ""): _mapping(row)
        for row in _sequence(certificate.get("actions"))
        if _mapping(row).get("action_ref")
    }
    residual_types = [
        str(row.get("residual_type"))
        for row in residuals_by_ref.values()
        if row.get("residual_type")
    ]
    leakage_refs = structured_certificate_leakage_refs(certificate)
    blocked_reasons: list[str] = []
    if leakage_refs:
        blocked_reasons.append("forbidden_runtime_reference")
    if "unknown_structured_residual" in set(residual_types):
        blocked_reasons.append("unknown_structured_residual")
    if not residual_types:
        blocked_reasons.append("no_open_structured_residual")

    bounded_steps: list[dict[str, Any]] = []
    for step in _sequence(certificate.get("repair_plan")):
        step_map = _mapping(step)
        stage = str(step_map.get("stage") or "")
        if stage not in REPAIR_STAGE_ORDER:
            blocked_reasons.append("unknown_repair_stage")
            continue
        step_residuals = [
            residuals_by_ref[ref]
            for ref in [str(item) for item in _sequence(step_map.get("residual_refs"))]
            if ref in residuals_by_ref
        ]
        step_constraints = [
            constraints_by_ref[ref]
            for ref in [
                str(item) for item in _sequence(step_map.get("constraint_refs"))
            ]
            if ref in constraints_by_ref
        ]
        action = actions_by_ref.get(str(step_map.get("action_ref") or ""), {})
        bounded_steps.append(
            {
                "step_ref": str(step_map.get("step_ref") or ""),
                "stage": stage,
                "stage_order_index": REPAIR_STAGE_ORDER.index(stage),
                "required_side": str(step_map.get("required_side") or "assistant"),
                "action_type": str(action.get("action_type") or ""),
                "residual_types": [
                    str(row.get("residual_type"))
                    for row in step_residuals
                    if row.get("residual_type")
                ],
                "constraint_types": [
                    str(row.get("constraint_type"))
                    for row in step_constraints
                    if row.get("constraint_type")
                ],
                "evidence_states": [
                    str(row.get("evidence_state"))
                    for row in step_residuals
                    if row.get("evidence_state")
                ],
                "status": str(step_map.get("status") or "open"),
            }
        )
    if residual_types and not bounded_steps:
        blocked_reasons.append("no_bounded_repair_step")

    action_obligations = action_obligations_from_structured_certificate(
        certificate,
        residual_types=residual_types,
        residual_vocabulary=residual_vocabulary,
        residual_to_operation_family=residual_to_operation_family,
        residual_to_argument_source=residual_to_argument_source,
        residual_to_target_selection_rule=residual_to_target_selection_rule,
        residual_to_completion_check=residual_to_completion_check,
        obligation_count_state=obligation_count_state,
        obligation_ordering=obligation_ordering,
    )
    abstract_templates = abstract_action_templates(action_obligations)
    ordered_stages = [step["stage"] for step in bounded_steps]
    return {
        "schema": STRUCTURED_REPAIR_DELTA_SCHEMA,
        "policy": STRUCTURED_REPAIR_POLICY,
        "action_obligation_policy": STRUCTURED_ACTION_OBLIGATION_POLICY,
        "eligible": not blocked_reasons,
        "blocked_reasons": sorted(set(blocked_reasons)),
        "repair_order": list(REPAIR_STAGE_ORDER),
        "ordered_stages": ordered_stages,
        "bounded_steps": bounded_steps,
        "action_obligations": action_obligations,
        "abstract_action_template": abstract_templates,
        "residual_types": [
            residual_type
            for residual_type in _residual_vocabulary(residual_vocabulary)
            if residual_type in set(residual_types)
        ],
        "max_delta_steps": len(bounded_steps),
        "verifier_scope": "bounded_delta_only",
        "forbidden_scope": [
            "do_not_full_resolve_task",
            "do_not_use_task_identifiers_or_hidden_comparator_outputs",
            "do_not_invent_concrete_entity_ids_or_fixed_tool_arguments",
            "do_not_repair_residuals_outside_bounded_steps",
        ],
        "leakage_refs": leakage_refs,
    }


def structured_repair_verifier_packet(
    certificate: Mapping[str, Any],
    *,
    residual_vocabulary: Sequence[str] | None = None,
    residual_to_operation_family: Mapping[str, str] | None = None,
    residual_to_argument_source: Mapping[str, str] | None = None,
    residual_to_target_selection_rule: Mapping[str, str] | None = None,
    residual_to_completion_check: Mapping[str, str] | None = None,
    obligation_count_state: Callable[[str, Mapping[str, Any]], dict[str, int]]
    | None = None,
    obligation_ordering: Callable[[str, Sequence[str]], list[str]] | None = None,
) -> dict[str, Any]:
    """Build a model-visible verifier packet for a bounded structured repair."""

    delta = structured_repair_delta(
        certificate,
        residual_vocabulary=residual_vocabulary,
        residual_to_operation_family=residual_to_operation_family,
        residual_to_argument_source=residual_to_argument_source,
        residual_to_target_selection_rule=residual_to_target_selection_rule,
        residual_to_completion_check=residual_to_completion_check,
        obligation_count_state=obligation_count_state,
        obligation_ordering=obligation_ordering,
    )
    packet = {
        "schema": STRUCTURED_REPAIR_PACKET_SCHEMA,
        "policy": STRUCTURED_REPAIR_POLICY,
        "structured_certificate": model_visible_structured_certificate(certificate),
        "bounded_delta": delta,
        "verifier_rule": (
            "Verify only the ordered bounded_delta steps. Approve when visible "
            "evidence already satisfies the listed residuals and constraints. "
            "When repairing, follow only the listed action_obligations and "
            "abstract_action_template entries. Do not solve the full task, "
            "invent concrete handles, or invent fixed tool arguments."
        ),
        "allowed_verdicts": ["approve", "repair", "stop"],
    }
    leakage_refs = find_forbidden_runtime_references(packet)
    packet["leakage_audit"] = {
        "audited": True,
        "leaked": bool(leakage_refs),
        "forbidden_paths": leakage_refs,
        "policy": "papc_structured_repair_packet_no_runtime_leakage_v1",
    }
    return packet


def action_obligations_from_structured_certificate(
    certificate: Mapping[str, Any],
    *,
    residual_types: Sequence[str] | None = None,
    residual_vocabulary: Sequence[str] | None = None,
    residual_to_operation_family: Mapping[str, str] | None = None,
    residual_to_argument_source: Mapping[str, str] | None = None,
    residual_to_target_selection_rule: Mapping[str, str] | None = None,
    residual_to_completion_check: Mapping[str, str] | None = None,
    obligation_count_state: Callable[[str, Mapping[str, Any]], dict[str, int]]
    | None = None,
    obligation_ordering: Callable[[str, Sequence[str]], list[str]] | None = None,
) -> list[dict[str, Any]]:
    """Build leakage-safe executable obligations from abstract residuals."""

    ordered_residuals = _ordered_unique_residual_types(
        residual_types
        if residual_types is not None
        else [
            str(_mapping(row).get("residual_type") or "")
            for row in _sequence(certificate.get("residuals"))
        ],
        residual_vocabulary=residual_vocabulary,
    )
    operation_map = _merge_residual_mapping(
        RESIDUAL_TO_OPERATION_FAMILY,
        residual_to_operation_family,
    )
    argument_source_map = _merge_residual_mapping(
        RESIDUAL_TO_ARGUMENT_SOURCE,
        residual_to_argument_source,
    )
    target_selection_map = _merge_residual_mapping(
        RESIDUAL_TO_TARGET_SELECTION_RULE,
        residual_to_target_selection_rule,
    )
    completion_map = _merge_residual_mapping(
        RESIDUAL_TO_COMPLETION_CHECK,
        residual_to_completion_check,
    )
    count_state = obligation_count_state or _obligation_count_state
    ordering = obligation_ordering or _obligation_ordering
    obligations: list[dict[str, Any]] = []
    for residual_type in ordered_residuals:
        counts = count_state(residual_type, certificate)
        obligation = {
            "operation_family": operation_map.get(
                residual_type,
                "adapter_mapping_review",
            ),
            "target_set_ref": "target_set:primary",
            "required_count": counts["required_count"],
            "observed_count": counts["observed_count"],
            "completed_count": counts["completed_count"],
            "missing_count": counts["missing_count"],
            "argument_source": argument_source_map.get(
                residual_type,
                "adapter_mapping_review",
            ),
            "target_selection_rule": target_selection_map.get(
                residual_type,
                "adapter_mapping_review",
            ),
            "ordering": ordering(residual_type, ordered_residuals),
            "completion_check": completion_map.get(
                residual_type,
                "adapter_mapping_reviewed",
            ),
        }
        obligations.append(_json_safe(obligation))
    return obligations


def abstract_action_templates(
    action_obligations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse obligations into model-visible abstract execution templates."""

    templates: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, tuple[str, ...]]] = set()
    for obligation in action_obligations:
        ordering = tuple(str(item) for item in _sequence(obligation.get("ordering")))
        template = {
            "operation_family": str(obligation.get("operation_family") or ""),
            "argument_source": str(obligation.get("argument_source") or ""),
            "target_selection_rule": str(obligation.get("target_selection_rule") or ""),
            "ordering": list(ordering),
            "completion_check": str(obligation.get("completion_check") or ""),
        }
        key = (
            template["operation_family"],
            template["argument_source"],
            template["target_selection_rule"],
            template["completion_check"],
            ordering,
        )
        if key in seen:
            continue
        seen.add(key)
        templates.append(template)
    return templates


def _obligation_count_state(
    residual_type: str,
    certificate: Mapping[str, Any],
) -> dict[str, int]:
    legacy = _mapping(certificate.get("legacy_projection"))
    target = _mapping(legacy.get("target_binding"))
    argument = _mapping(legacy.get("argument_binding"))
    commit = _mapping(legacy.get("commit_contract"))
    terminal_plan = _mapping(legacy.get("terminal_completion_plan"))
    structured_targets = _sequence(certificate.get("targets"))
    count_state = {}
    if structured_targets:
        count_state = _mapping(_mapping(structured_targets[0]).get("count_state"))

    if residual_type == "terminal_completion":
        required = max(
            _int_value(commit.get("expected_write_count")),
            _int_value(terminal_plan.get("expected_write_count")),
            _int_value(count_state.get("expected_write_count")),
        )
        observed = max(
            _int_value(target.get("observed_count")),
            _int_value(count_state.get("observed_count")),
        )
        completed = max(
            _int_value(commit.get("completed_write_count")),
            _int_value(terminal_plan.get("completed_write_count")),
            _int_value(count_state.get("completed_count")),
        )
    elif residual_type in {
        "argument_binding",
        "argument_preference_binding",
        "write_argument_binding",
        "policy_boundary",
    }:
        required = max(
            1,
            _int_value(argument.get("argument_count")),
            _int_value(argument.get("write_argument_count")),
        )
        observed = max(required, _int_value(target.get("observed_count")))
        completed = _int_value(argument.get("completed_binding_count"))
    else:
        required = max(
            _int_value(target.get("target_count")),
            _int_value(target.get("observed_count")),
            _int_value(count_state.get("candidate_count")),
            _int_value(count_state.get("observed_count")),
        )
        observed = max(
            _int_value(target.get("observed_count")),
            _int_value(count_state.get("observed_count")),
            required,
        )
        completed = max(
            _int_value(target.get("completed_write_count")),
            _int_value(commit.get("completed_write_count")),
            _int_value(count_state.get("completed_count")),
        )

    if required <= 0 and residual_type:
        required = 1
    if observed <= 0:
        observed = required
    completed = max(0, min(completed, max(required, observed)))
    return {
        "required_count": required,
        "observed_count": observed,
        "completed_count": completed,
        "missing_count": max(required - completed, 0),
    }


def _obligation_ordering(
    residual_type: str,
    residual_types: Sequence[str],
) -> list[str]:
    residual_set = {str(item) for item in residual_types if str(item)}
    ordering: list[str] = []
    if residual_type == "terminal_completion" and (
        "evidence_sweep" in residual_set
        or "target_set_binding" in residual_set
        or "source_read_binding" in residual_set
        or "selection_target_binding" in residual_set
    ):
        ordering.append("evidence_and_target_binding_before_terminal_write")
    if residual_type == "terminal_completion" and (
        "argument_binding" in residual_set
        or "write_argument_binding" in residual_set
        or "argument_preference_binding" in residual_set
    ):
        ordering.append("argument_binding_before_terminal_write")
    if residual_type == "postcondition_risk":
        ordering.append("postcondition_check_after_candidate_write_surface")
    return ordering


def _status_reason_to_residual(reason: str) -> str:
    if "precondition" in reason or "auth" in reason:
        return "precondition_binding"
    if "source" in reason or "read" in reason:
        return "source_read_binding"
    if "selection" in reason:
        return "selection_target_binding"
    if "target" in reason:
        return "target_set_binding"
    if "evidence" in reason or "lookup" in reason:
        return "evidence_sweep"
    if "write_argument" in reason:
        return "write_argument_binding"
    if "argument" in reason or "binding" in reason:
        return "argument_binding"
    if "preference" in reason or "choice" in reason:
        return "argument_preference_binding"
    if "terminal" in reason or "write" in reason:
        return "terminal_completion"
    if "postcondition" in reason or "collateral" in reason or "duplicate" in reason:
        return "postcondition_risk"
    if "policy" in reason:
        return "policy_boundary"
    return "unknown_structured_residual"


def _add_status_residual(
    residual_types: list[str],
    source_reasons: dict[str, list[str]],
    *,
    residual_type: str,
    source: str,
    value: Any,
    bad_values: set[str],
) -> None:
    text = str(value or "")
    if text not in bad_values:
        return
    residual_types.append(residual_type)
    source_reasons.setdefault(residual_type, []).append(f"{source}:{text}")


def _residual_evidence_state(
    residual_type: str,
    certificate: Mapping[str, Any],
) -> str:
    precondition = _mapping(certificate.get("precondition_state"))
    target = _mapping(certificate.get("target_binding"))
    argument = _mapping(certificate.get("argument_binding"))
    postcondition = _mapping(certificate.get("postcondition"))
    if residual_type == "evidence_sweep":
        return str(precondition.get("tool_evidence_status") or "open")
    if residual_type == "target_set_binding":
        return str(
            target.get("target_binding_status")
            or target.get("target_sweep_status")
            or "open"
        )
    if residual_type == "precondition_binding":
        return str(precondition.get("precondition_status") or "open")
    if residual_type == "source_read_binding":
        return str(precondition.get("source_read_status") or "open")
    if residual_type == "selection_target_binding":
        return str(target.get("selection_binding_status") or "open")
    if "argument" in residual_type or "preference" in residual_type:
        return str(argument.get("argument_binding_status") or "open")
    if residual_type == "terminal_completion":
        return str(postcondition.get("terminal_completion_status") or "open")
    if residual_type == "postcondition_risk":
        return str(postcondition.get("postcondition_status") or "open")
    return "open"


def _severity_by_type(
    level: Any,
    *,
    residual_severity_overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    fallback = str(level or "medium")
    severity = {
        "postcondition_risk": "high",
        "terminal_completion": "high",
        "precondition_binding": "high",
        "target_set_binding": fallback,
        "source_read_binding": fallback,
        "selection_target_binding": fallback,
        "argument_binding": fallback,
        "write_argument_binding": fallback,
        "argument_preference_binding": fallback,
        "evidence_sweep": fallback,
        "policy_boundary": "high",
        "unknown_structured_residual": "medium",
    }
    severity.update(
        {
            str(key): str(value)
            for key, value in dict(residual_severity_overrides or {}).items()
        }
    )
    return severity


def _ordered_unique_residual_types(
    residual_types: Sequence[str],
    *,
    residual_vocabulary: Sequence[str] | None = None,
) -> list[str]:
    residual_set = {str(item) for item in residual_types if str(item)}
    vocabulary = _residual_vocabulary(residual_vocabulary)
    ordered = [item for item in vocabulary if item in residual_set]
    extras = sorted(residual_set - set(vocabulary))
    return ordered + extras


def _residual_vocabulary(
    residual_vocabulary: Sequence[str] | None = None,
) -> tuple[str, ...]:
    if residual_vocabulary is None:
        return GENERIC_RESIDUAL_TYPES
    seen: dict[str, None] = {}
    for residual_type in (
        *[str(item) for item in residual_vocabulary],
        *GENERIC_RESIDUAL_TYPES,
    ):
        if residual_type:
            seen.setdefault(residual_type, None)
    return tuple(seen)


def _merge_residual_mapping(
    base: Mapping[str, str],
    override: Mapping[str, str] | None,
) -> dict[str, str]:
    merged = {str(key): str(value) for key, value in base.items()}
    merged.update({str(key): str(value) for key, value in dict(override or {}).items()})
    return merged


def _action_type_for_stage(stage: str) -> str:
    return {
        "evidence": "gather_visible_evidence",
        "user_choice": "bind_user_side_choice",
        "argument_binding": "repair_argument_binding",
        "terminal_write": "complete_terminal_write_set",
        "postcondition": "verify_postcondition",
    }.get(stage, "adapter_mapping_review")


def _required_side_for_stage(stage: str) -> str:
    return "user" if stage == "user_choice" else "assistant"


def _target_scope(target: Mapping[str, Any], commit: Mapping[str, Any]) -> str:
    if target.get("target_scope"):
        return str(target.get("target_scope"))
    if commit.get("plural_target_scope"):
        return "plural_target_scope"
    count = _int_value(target.get("target_count"))
    if count > 1:
        return "plural_target_scope"
    if count == 1:
        return "single_target_scope"
    return "unknown_scope"


def _first_nonempty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return "unknown"


def _compact_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _json_safe(item)
        for key, item in value.items()
        if item not in (None, "", [], {})
    }


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in _sequence(value) if str(item)]


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_safe(item) for item in value]
    return value

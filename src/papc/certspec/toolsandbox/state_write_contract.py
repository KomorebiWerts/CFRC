"""ToolSandbox projection into PAPC state-write contracts.

The ToolSandbox runtime adapter must not read scenario names, task ids, gold
milestones, expected arguments, raw prompts, or hidden snapshots.  This module
therefore consumes only the sanitized runtime digest shape produced by the
no-API launch guard: visible prefix counts, observed tool-state counts, and
abstract residual requirements such as dependency closure and minefield
avoidance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from papc.certspec.reversibility import (
    DEFAULT_RESTORATION_COST,
    IRREVERSIBLE_RESTORATION_COST,
)
from papc.certspec.state_write_contract import (
    ContractJudgment,
    Obligation,
    ProposedTargetAction,
    StateWriteContract,
    idempotency_precheck,
    judge_target_action,
    project_admissible_action,
)

SURFACE = "toolsandbox"

DEPENDENCY_COMPLETION_UNIT = "toolsandbox.dependency_completion"
MINEFIELD_SIDE_EFFECT_UNIT = "toolsandbox.minefield_side_effect"
OBSERVED_STATE_UNIT_PREFIX = "toolsandbox.observed_state"

LEGAL_EFFECT_UNITS = frozenset({DEPENDENCY_COMPLETION_UNIT})
FORBIDDEN_EFFECT_UNITS = frozenset({MINEFIELD_SIDE_EFFECT_UNIT})


def state_write_contract_from_runtime_digest(
    digest: Mapping[str, Any],
) -> StateWriteContract:
    """Build a ToolSandbox contract from a sanitized runtime digest.

    ``observed_state`` entries become read-only dependency evidence.  They are
    deliberately not placed in ``forbidden_writes`` because reads are not
    commitments.  Minefield avoidance, by contrast, is a forbidden side-effect
    unit: a proposal that writes it must be rejected or projected away.
    """

    residual_request = _mapping(digest.get("residual_request"))
    requires_dependency = bool(residual_request.get("requires_dependency_closure"))
    requires_minefield_avoidance = bool(
        residual_request.get("requires_illegal_side_effect_avoidance")
    )

    residual: set[Obligation] = set()
    if requires_dependency:
        residual.add(
            Obligation(
                obligation_id="toolsandbox.residual.dependency_completion",
                write_footprint=frozenset({DEPENDENCY_COMPLETION_UNIT}),
                postcondition_witness="visible_prefix_tool_state_residual_request",
                surface=SURFACE,
                status="open_bounded",
            )
        )

    forbidden = (
        frozenset({MINEFIELD_SIDE_EFFECT_UNIT})
        if requires_minefield_avoidance
        else frozenset()
    )
    observed_reads = _observed_state_units(digest)
    residual_units = frozenset(
        unit for obligation in residual for unit in obligation.write_footprint
    )
    legal_actions = (
        frozenset({"bounded_repair", "abstain"})
        if residual_units
        else frozenset({"accept", "abstain"})
    )

    return StateWriteContract(
        surface=SURFACE,
        derivation_timing="runtime_progressive",
        residual=frozenset(residual),
        forbidden_writes=forbidden,
        legal_actions=legal_actions,
        state_change_required=bool(residual_units),
        commitment_call_count=0,
        commitment_footprint_size=len(forbidden),
        commitment_app_count=0,
        soundness_assumptions=(
            "toolsandbox_contract_uses_sanitized_runtime_digest_only",
            "observed_tool_state_is_read_evidence_not_committed_write",
            "minefield_avoidance_is_modelled_as_forbidden_side_effect",
        ),
        observed_reads=observed_reads,
        touched_units=observed_reads | residual_units | forbidden,
    )


def proposed_target_action_from_probe(
    proposal: Mapping[str, Any],
) -> ProposedTargetAction:
    """Project an abstract ToolSandbox probe proposal into target effects."""

    raw_calls = proposal.get("tool_calls")
    calls = raw_calls if isinstance(raw_calls, Sequence) and not isinstance(raw_calls, str) else ()
    effect_groups: list[frozenset[str]] = []
    effects: set[str] = set()
    for call in calls:
        if not isinstance(call, Mapping):
            continue
        group = _call_effect_units(call)
        if not group:
            continue
        effect_groups.append(group)
        effects.update(group)

    action = str(proposal.get("action") or "execute")
    assistant_message = str(proposal.get("assistant_message") or "")
    return ProposedTargetAction(
        surface=SURFACE,
        action_id=f"toolsandbox_probe:{action}:{len(effect_groups)}groups",
        write_footprint=frozenset(effects),
        terminal_completion=bool(action == "execute" and assistant_message and not effects),
        effect_groups=tuple(effect_groups),
    )


def restoration_cost_vector(units: Sequence[str]) -> dict[str, float]:
    """Return schema-level restoration costs for abstract ToolSandbox units."""

    return {
        unit: (
            IRREVERSIBLE_RESTORATION_COST
            if _unit_family(unit) == MINEFIELD_SIDE_EFFECT_UNIT
            else DEFAULT_RESTORATION_COST
        )
        for unit in units
    }


def kernel_verdict_json(
    *,
    digest: Mapping[str, Any],
    proposal: Mapping[str, Any],
    collision_budget: float = 0.0,
    require_full_discharge: bool = False,
) -> dict[str, Any]:
    """Return the same receipt-level kernel shape required from online runners."""

    contract = state_write_contract_from_runtime_digest(digest)
    action = proposed_target_action_from_probe(proposal)
    cost_vector = restoration_cost_vector(sorted(contract.touched_units | action.effects()))
    idempotency = idempotency_precheck(contract, action)
    judgment = judge_target_action(contract, action)
    projection = project_admissible_action(
        contract,
        action,
        restoration_cost=cost_vector,
        collision_budget=collision_budget,
        require_full_discharge=require_full_discharge,
    )
    return {
        "schema": "papc.toolsandbox.kernel_verdict.v1",
        "surface": SURFACE,
        "restoration_cost_mode": "schema_restoration_cost",
        "blocks_target": idempotency.blocks_target or judgment.blocks_target,
        "idempotency_precheck": idempotency.to_json(),
        "contract_judgment": judgment.to_json(),
        "action_projection": projection.to_json(),
        "state_write_contract": contract.to_json(),
        "proposed_target_action": action.to_json(),
    }


def q_precheck_judgment_from_runtime_digest(
    digest: Mapping[str, Any],
    proposal: Mapping[str, Any],
) -> ContractJudgment:
    """Compile the ToolSandbox runtime digest into a protocol judgment."""

    contract = state_write_contract_from_runtime_digest(digest)
    action = proposed_target_action_from_probe(proposal)
    if contract.has_residual_state_change() and not action.write_footprint:
        return ContractJudgment(
            decision="block_q_precheck",
            reason="toolsandbox_dependency_completion_requires_tool_effect",
            enforcement_point="q_precheck",
        )
    return ContractJudgment(
        decision="allow",
        reason="toolsandbox_runtime_digest_q_precheck_pass",
        enforcement_point="q_precheck",
        allowed_footprint=action.write_footprint,
    )


def _observed_state_units(digest: Mapping[str, Any]) -> frozenset[str]:
    observed_state = _mapping(digest.get("observed_state"))
    count = _nonnegative_int(observed_state.get("observed_state_units"))
    return frozenset(f"{OBSERVED_STATE_UNIT_PREFIX}#{index}" for index in range(count))


def _call_effect_units(call: Mapping[str, Any]) -> frozenset[str]:
    units: set[str] = set()
    raw_units = call.get("effect_units") or call.get("effect_families")
    if isinstance(raw_units, Sequence) and not isinstance(raw_units, str):
        for unit in raw_units:
            normalized = _normalize_effect_unit(str(unit))
            if normalized:
                units.add(normalized)
    normalized = _normalize_effect_unit(str(call.get("effect_family") or ""))
    if normalized:
        units.add(normalized)
    if call.get("writes_dependency_completion") is True:
        units.add(DEPENDENCY_COMPLETION_UNIT)
    if call.get("triggers_minefield") is True:
        units.add(MINEFIELD_SIDE_EFFECT_UNIT)
    name = str(call.get("name") or "").lower()
    if "dependency" in name or "milestone" in name:
        units.add(DEPENDENCY_COMPLETION_UNIT)
    if "minefield" in name or "illegal" in name:
        units.add(MINEFIELD_SIDE_EFFECT_UNIT)
    return frozenset(units)


def _normalize_effect_unit(raw: str) -> str | None:
    value = raw.strip().lower()
    if value in {
        DEPENDENCY_COMPLETION_UNIT,
        "dependency_completion",
        "dependency",
        "milestone",
        "milestone_completion",
    }:
        return DEPENDENCY_COMPLETION_UNIT
    if value in {
        MINEFIELD_SIDE_EFFECT_UNIT,
        "minefield_side_effect",
        "minefield",
        "illegal_side_effect",
    }:
        return MINEFIELD_SIDE_EFFECT_UNIT
    return None


def _unit_family(unit: str) -> str:
    return unit.split("#", 1)[0]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


__all__ = [
    "SURFACE",
    "DEPENDENCY_COMPLETION_UNIT",
    "MINEFIELD_SIDE_EFFECT_UNIT",
    "LEGAL_EFFECT_UNITS",
    "FORBIDDEN_EFFECT_UNITS",
    "state_write_contract_from_runtime_digest",
    "proposed_target_action_from_probe",
    "restoration_cost_vector",
    "kernel_verdict_json",
    "q_precheck_judgment_from_runtime_digest",
]

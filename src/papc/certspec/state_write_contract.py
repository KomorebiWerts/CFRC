"""Shared state-write contract primitives for non-regressive escalation.

This module is benchmark-independent. Dataset adapters may project visible
tool traces into these objects, but this core must not inspect dataset names,
task ids, gold outcomes, concrete entity ids, or fixed tool arguments.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

ContractAction = Literal["accept", "bounded_repair", "abstain", "fallback"]
JudgmentDecision = Literal[
    "allow",
    "allow_bounded",
    "block_closed_pre_target",
    "block_missing_required_write",
    "block_no_progress",
    "block_overwrite",
    "block_duplicate_write",
    "block_irreversible_collision",
    "block_q_precheck",
    "abstain",
]
EnforcementPoint = Literal[
    "pre_target_veto",
    "q_precheck",
    "post_target_validator",
    "sandbox_validator",
    "post_commit_compensator",
]
ProjectionSolver = Literal["auto", "greedy", "exact_enumeration"]


@dataclass(frozen=True)
class Obligation:
    """A closed or residual state obligation projected from visible evidence."""

    obligation_id: str
    write_footprint: frozenset[str] = field(default_factory=frozenset)
    postcondition_witness: str = ""
    surface: str = ""
    status: Literal["closed", "open_bounded", "open_unbounded"] = "closed"

    def req(self) -> frozenset[str]:
        """Required state units ``req(o)`` for satisfying this obligation."""

        return self.write_footprint

    def to_json(self) -> dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "write_footprint": sorted(self.write_footprint),
            "required_footprint": sorted(self.req()),
            "postcondition_witness": self.postcondition_witness,
            "surface": self.surface,
            "status": self.status,
        }


@dataclass(frozen=True)
class StateWriteContract:
    """Runtime-visible contract over closed and residual state writes."""

    surface: str
    derivation_timing: Literal["prefix", "runtime_progressive", "terminal"]
    closed: frozenset[Obligation] = field(default_factory=frozenset)
    residual: frozenset[Obligation] = field(default_factory=frozenset)
    forbidden_writes: frozenset[str] = field(default_factory=frozenset)
    legal_actions: frozenset[ContractAction] = field(default_factory=frozenset)
    state_change_required: bool = False
    user_facing_message_required: bool = False
    commitment_call_count: int = 0
    commitment_footprint_size: int = 0
    commitment_app_count: int = 0
    soundness_assumptions: tuple[str, ...] = ()
    # D(z): dependency-bearing observed reads. Kept SEPARATE from the committed
    # set C(z) (forbidden_writes + closed-obligation footprints) so a plain
    # verified read never blocks a later overwrite; used only for stale-
    # precondition detection, never for the collision constraint.
    observed_reads: frozenset[str] = field(default_factory=frozenset)
    # T(z): runtime-touched state units, tracked as metadata for surface-card
    # auditing. This is not C(z), not R(z), and not a collision mask input.
    touched_units: frozenset[str] = field(default_factory=frozenset)

    def has_closed_state(self) -> bool:
        return bool(self.closed or self.forbidden_writes)

    def has_residual_state_change(self) -> bool:
        return bool(self.residual) and self.state_change_required

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "papc.state_write_contract.v1",
            "surface": self.surface,
            "derivation_timing": self.derivation_timing,
            "closed": [
                item.to_json() for item in sorted(self.closed, key=_obligation_key)
            ],
            "residual": [
                item.to_json() for item in sorted(self.residual, key=_obligation_key)
            ],
            "forbidden_writes": sorted(self.forbidden_writes),
            "legal_actions": sorted(self.legal_actions),
            "state_change_required": self.state_change_required,
            "user_facing_message_required": self.user_facing_message_required,
            "commitment_volume": {
                "state_changing_call_count": self.commitment_call_count,
                "forbidden_footprint_size": self.commitment_footprint_size,
                "app_count": self.commitment_app_count,
            },
            "soundness_assumptions": list(self.soundness_assumptions),
            "observed_reads": sorted(self.observed_reads),
            "touched_units": sorted(self.touched_units),
        }


@dataclass(frozen=True)
class ProposedTargetAction:
    """Abstract target action proposed by an escalation policy.

    ``write_footprint`` is the proposed *effect* footprint ``E(a)`` -- the units
    the action would actually write -- NOT the legal authority envelope. The
    optional ``writable_scope`` is that envelope ``W(a)`` (the upper bound on
    units the authority is permitted to modify); when declared, the effects must
    lie within it (``E(a) subset W(a)``). ``effect_groups`` and ``dependencies``
    declare the dependency-closure structure ``F(a,z)``: atomic effect groups
    that must be kept or trimmed together, and prerequisite edges
    ``unit -> required units``, so :func:`project_admissible_action` cannot keep
    an incoherent subset of a transactional write (e.g. ``attach_item`` without
    ``create_order``). All three are empty by default, in which case the
    projection is the flat per-unit projection over ``E(a)``.
    """

    surface: str
    action_id: str
    write_footprint: frozenset[str] = field(default_factory=frozenset)
    read_footprint: frozenset[str] = field(default_factory=frozenset)
    terminal_completion: bool = False
    execution_succeeded: bool | None = None
    writable_scope: frozenset[str] = field(default_factory=frozenset)
    effect_groups: tuple[frozenset[str], ...] = ()
    dependencies: tuple[tuple[str, frozenset[str]], ...] = ()
    discharged_obligation_ids: frozenset[str] = field(default_factory=frozenset)

    def effects(self) -> frozenset[str]:
        """Proposed effect footprint ``E(a)`` (the units actually written)."""

        return self.write_footprint

    def authority_envelope(self) -> frozenset[str]:
        """Legal authority envelope ``W(a)``; defaults to ``E(a)`` when undeclared."""

        return self.writable_scope or self.write_footprint

    def within_authority(self) -> bool:
        """Whether the proposed effects respect the declared authority envelope."""

        return not self.writable_scope or self.write_footprint <= self.writable_scope

    def dependency_map(self) -> dict[str, frozenset[str]]:
        """Prerequisite edges as ``unit -> required prerequisite units``."""

        return {unit: frozenset(prereqs) for unit, prereqs in self.dependencies}

    def has_closure_structure(self) -> bool:
        """Whether any dependency-closure structure ``F(a,z)`` is declared."""

        return bool(self.effect_groups or self.dependencies)

    def to_json(self) -> dict[str, Any]:
        payload = {
            "schema": "papc.proposed_target_action.v1",
            "surface": self.surface,
            "action_id": self.action_id,
            "write_footprint": sorted(self.write_footprint),
            "read_footprint": sorted(self.read_footprint),
            "terminal_completion": self.terminal_completion,
            "execution_succeeded": self.execution_succeeded,
            "writable_scope": sorted(self.writable_scope),
            "effect_groups": [sorted(group) for group in self.effect_groups],
            "dependencies": [
                [unit, sorted(prereqs)] for unit, prereqs in self.dependencies
            ],
        }
        if self.discharged_obligation_ids:
            payload["discharged_obligation_ids"] = sorted(
                self.discharged_obligation_ids
            )
        return payload


@dataclass(frozen=True)
class ContractJudgment:
    """Decision made by a contract enforcement point."""

    decision: JudgmentDecision
    reason: str
    enforcement_point: EnforcementPoint
    blocked_footprint: frozenset[str] = field(default_factory=frozenset)
    allowed_footprint: frozenset[str] = field(default_factory=frozenset)

    @property
    def blocks_target(self) -> bool:
        return self.decision.startswith("block") or self.decision == "abstain"

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "papc.contract_judgment.v1",
            "decision": self.decision,
            "reason": self.reason,
            "enforcement_point": self.enforcement_point,
            "blocks_target": self.blocks_target,
            "blocked_footprint": sorted(self.blocked_footprint),
            "allowed_footprint": sorted(self.allowed_footprint),
        }


def pre_target_veto(contract: StateWriteContract) -> ContractJudgment:
    """Decide whether a target call is legal before spending the target forward."""

    if contract.has_closed_state() and not contract.has_residual_state_change():
        return ContractJudgment(
            decision="block_closed_pre_target",
            reason="closed_obligations_have_no_visible_residual",
            enforcement_point="pre_target_veto",
            blocked_footprint=contract.forbidden_writes,
        )
    if "abstain" in contract.legal_actions and not contract.legal_actions & {
        "bounded_repair"
    }:
        return ContractJudgment(
            decision="abstain",
            reason="contract_exposes_no_legal_target_repair_action",
            enforcement_point="pre_target_veto",
        )
    return ContractJudgment(
        decision="allow",
        reason="target_call_allowed_by_contract",
        enforcement_point="pre_target_veto",
        allowed_footprint=contract.forbidden_writes,
    )


def judge_target_action(
    contract: StateWriteContract,
    proposed_action: ProposedTargetAction,
    *,
    enforcement_point: EnforcementPoint = "post_target_validator",
) -> ContractJudgment:
    """Judge whether a target proposal respects the state-write contract."""

    if not proposed_action.within_authority():
        outside_authority = (
            proposed_action.write_footprint - proposed_action.authority_envelope()
        )
        return ContractJudgment(
            decision="block_missing_required_write",
            reason="target_proposal_effects_escape_authority_envelope",
            enforcement_point=enforcement_point,
            blocked_footprint=outside_authority,
            allowed_footprint=proposed_action.write_footprint - outside_authority,
        )

    blocked = contract.forbidden_writes & proposed_action.write_footprint
    if blocked:
        return ContractJudgment(
            decision="block_overwrite",
            reason="target_proposal_touches_forbidden_closed_footprint",
            enforcement_point=enforcement_point,
            blocked_footprint=blocked,
        )

    residual_write_footprint = _residual_write_footprint(contract)
    if contract.has_residual_state_change() and not proposed_action.write_footprint:
        reason = (
            "terminal_completion_without_required_state_write"
            if proposed_action.terminal_completion
            else "target_proposal_has_no_bounded_state_write"
        )
        return ContractJudgment(
            decision=(
                "block_missing_required_write"
                if proposed_action.terminal_completion
                else "block_no_progress"
            ),
            reason=reason,
            enforcement_point=enforcement_point,
        )

    if proposed_action.write_footprint:
        if contract.has_residual_state_change() and residual_write_footprint:
            allowed = proposed_action.write_footprint & residual_write_footprint
            outside_residual = (
                proposed_action.write_footprint - residual_write_footprint
            )
            if outside_residual or not allowed:
                return ContractJudgment(
                    decision="block_missing_required_write",
                    reason="target_proposal_writes_outside_legal_residual_footprint",
                    enforcement_point=enforcement_point,
                    blocked_footprint=outside_residual
                    or proposed_action.write_footprint,
                    allowed_footprint=allowed,
                )
            return ContractJudgment(
                decision="allow_bounded",
                reason="target_writes_hit_legal_residual_footprint",
                enforcement_point=enforcement_point,
                allowed_footprint=allowed,
            )
        if "bounded_repair" not in contract.legal_actions:
            return ContractJudgment(
                decision="block_missing_required_write",
                reason="contract_exposes_no_legal_residual_write_scope",
                enforcement_point=enforcement_point,
                blocked_footprint=proposed_action.write_footprint,
            )
        return ContractJudgment(
            decision="allow_bounded",
            reason="target_writes_are_outside_forbidden_closed_footprint",
            enforcement_point=enforcement_point,
            allowed_footprint=proposed_action.write_footprint,
        )

    return ContractJudgment(
        decision="allow",
        reason="no_residual_noop",
        enforcement_point=enforcement_point,
    )


DEFAULT_RESTORATION_COST = 1.0
IRREVERSIBLE_RESTORATION_COST = float("inf")


def restoration_weighted_collision(
    units: frozenset[str],
    restoration_cost: Mapping[str, float] | None = None,
) -> float:
    """Restoration-cost-weighted collision measure ``Lambda``.

    ``Lambda(units) = sum_u restoration_cost[u]`` over committed-state units a
    proposed write would touch. ``restoration_cost[u]`` is the declared expected
    cost to undo or compensate a corruption of unit ``u`` (from logged
    rollback/readback or tool-schema reversibility), NOT an oracle damage label.

    With the default ``restoration_cost = None`` every unit weighs
    ``DEFAULT_RESTORATION_COST = 1.0``, so ``Lambda`` reduces exactly to the
    binary counting measure ``mu`` used by the v1 mask. Schema-declared weights
    make truly irreversible units (an append-only delete, a sent payment) bind
    the budget while cheaply restorable units (a cart item) do not.
    """

    if restoration_cost is None:
        return float(len(units))
    return float(sum(restoration_cost.get(u, DEFAULT_RESTORATION_COST) for u in units))


def irreversible_collision_units(
    units: frozenset[str],
    restoration_cost: Mapping[str, float] | None = None,
) -> frozenset[str]:
    """Committed units that are typed as non-compensable damage.

    The hard block is only active when a surface explicitly declares a unit's
    restoration cost as ``IRREVERSIBLE_RESTORATION_COST``. Uniform-count and
    legacy callers that omit ``restoration_cost`` keep their old behavior.
    """

    if restoration_cost is None:
        return frozenset()
    return frozenset(
        unit
        for unit in units
        if math.isinf(restoration_cost.get(unit, DEFAULT_RESTORATION_COST))
    )


def idempotency_precheck(
    contract: StateWriteContract,
    proposed_action: ProposedTargetAction,
    *,
    enforcement_point: EnforcementPoint = "post_target_validator",
) -> ContractJudgment:
    """Reject a proposal that only repeats already committed writes.

    This is intentionally a separate precheck instead of changing
    :func:`judge_target_action`'s long-standing overwrite semantics. Closed-loop
    residual runners can call it before projection and emit a sharper
    ``block_duplicate_write`` reason without disturbing existing baselines.
    """

    duplicate = contract.forbidden_writes & proposed_action.write_footprint
    non_duplicate = proposed_action.write_footprint - duplicate
    if duplicate and not non_duplicate:
        return ContractJudgment(
            decision="block_duplicate_write",
            reason="target_proposal_repeats_only_committed_write_footprint",
            enforcement_point=enforcement_point,
            blocked_footprint=duplicate,
        )
    return ContractJudgment(
        decision="allow",
        reason="target_proposal_has_non_duplicate_effects",
        enforcement_point=enforcement_point,
        allowed_footprint=non_duplicate or proposed_action.write_footprint,
    )


@dataclass(frozen=True)
class ActionProjection:
    """Result of projecting a proposed action onto its admissible sub-action.

    This is the v2 generative core: instead of accepting or rejecting the whole
    proposed write footprint (the v1 all-or-nothing mask), the projection
    operator ``pi_C`` returns a feasible dependency-closed sub-footprint that
    respects the restoration-cost collision budget while still making residual
    progress. Small effect graphs are solved by exact enumeration; larger graphs
    fall back to deterministic greedy readmission and report that solver choice.
    """

    surface: str
    action_id: str
    proposed_footprint: frozenset[str] = field(default_factory=frozenset)
    projected_footprint: frozenset[str] = field(default_factory=frozenset)
    trimmed_footprint: frozenset[str] = field(default_factory=frozenset)
    readmitted_footprint: frozenset[str] = field(default_factory=frozenset)
    lambda_collision: float = 0.0
    collision_budget: float = 0.0
    progress_units: int = 0
    discharged_obligation_count: int = 0
    discharged_obligations: tuple[str, ...] = ()
    projection_claim: str = "feasible_dependency_closed_projection"
    readmission_rule: str = "greedy_ascending_restoration_cost"
    decision: Literal["project_bounded", "project_abstain", "project_identity"] = (
        "project_abstain"
    )

    @property
    def makes_progress(self) -> bool:
        return self.progress_units > 0 or self.discharged_obligation_count > 0

    @property
    def fully_discharges_obligation(self) -> bool:
        """Whether the projected action fully closes at least one ``R(z)`` obligation."""

        return self.discharged_obligation_count > 0

    @property
    def trims(self) -> bool:
        return bool(self.trimmed_footprint)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "papc.action_projection.v1",
            "surface": self.surface,
            "action_id": self.action_id,
            "proposed_footprint_size": len(self.proposed_footprint),
            "projected_footprint": sorted(self.projected_footprint),
            "projected_footprint_size": len(self.projected_footprint),
            "trimmed_footprint": sorted(self.trimmed_footprint),
            "trimmed_footprint_size": len(self.trimmed_footprint),
            "readmitted_footprint": sorted(self.readmitted_footprint),
            "lambda_collision": self.lambda_collision,
            "collision_budget": self.collision_budget,
            "progress_units": self.progress_units,
            "discharged_obligation_count": self.discharged_obligation_count,
            "discharged_obligations": list(self.discharged_obligations),
            "projection_claim": self.projection_claim,
            "readmission_rule": self.readmission_rule,
            "makes_progress": self.makes_progress,
            "trims": self.trims,
            "decision": self.decision,
        }


def project_admissible_action(
    contract: StateWriteContract,
    proposed_action: ProposedTargetAction,
    *,
    restoration_cost: Mapping[str, float] | None = None,
    collision_budget: float = 0.0,
    require_full_discharge: bool = False,
    solver: ProjectionSolver = "auto",
    exact_enumeration_limit: int = 16,
) -> ActionProjection:
    """Project a proposed action onto a feasible admissible sub-action ``pi_C``.

    Unlike :func:`judge_target_action`, which rejects the entire proposal when
    any unit collides with committed state, ``pi_C`` keeps the safe part. It
    ranges over the proposed *effect* footprint ``E(a) = proposed_action.effects()``
    (not the authority envelope ``W(a)``) and:

    1. drops every committed unit ``E(a)`` would overwrite
       (``C(z) = contract.forbidden_writes``);
    2. if residual obligations exist, keeps only units inside the legal residual
       write footprint (no writing outside the open residual scope);
    3. for small effect graphs, enumerates dependency-closed feasible
       sub-actions and chooses the one that maximizes discharged obligations
       and residual progress under the budget; for larger graphs, re-admits the
       cheapest-to-restore colliding units in ascending restoration cost while
       cumulative ``Lambda(E(a') cap C(z))`` stays within ``collision_budget``
       (default ``0.0`` re-admits nothing);
    4. enforces the declared dependency closure ``F(a,z)`` (atomic effect groups
       and prerequisite edges) so no incoherent subset survives -- a no-op when
       the proposal declares no closure structure, recovering the flat projection.

    ``solver="auto"`` uses exact enumeration for small graphs (bounded by
    ``exact_enumeration_limit``) and the deterministic greedy feasible fallback
    otherwise. Use ``solver="greedy"`` to recover the historical projection
    exactly, or ``solver="exact_enumeration"`` to require the exact small-graph
    path (raising ``ValueError`` when the proposal is too large).

    ``discharged_obligations`` is ``G(a') subset R(z)``: the residual obligations
    whose entire required footprint the projected action covers, union explicit
    effect-free discharges carried by a terminal message. Unknown ids and ids on
    a non-terminal action are ignored. Pass ``require_full_discharge=True`` to
    gate admissibility on a fully closed obligation rather than on the unit-level
    surrogate ``progress_units`` (retained for the v1 arm pipeline). The
    projection is the identity when nothing is trimmed, ``project_bounded`` when
    it makes residual progress, and ``project_abstain`` (equivalent to A5) when
    nothing admissible survives.
    """

    forbidden = contract.forbidden_writes
    legal_residual = _residual_write_footprint(contract)
    proposed = frozenset(proposed_action.effects())
    authority = proposed_action.authority_envelope()
    outside_authority = proposed - authority
    legal_proposed = proposed - outside_authority
    explicit_discharges = _explicit_discharged_obligations(
        contract,
        proposed_action,
    )
    irreversible_forbidden = irreversible_collision_units(
        legal_proposed & forbidden,
        restoration_cost,
    )
    legal_for_projection = legal_proposed - irreversible_forbidden
    budgetable_forbidden = forbidden - irreversible_forbidden

    if solver not in ("auto", "greedy", "exact_enumeration"):
        raise ValueError(f"unknown projection solver: {solver!r}")

    use_exact = solver == "exact_enumeration" or (
        solver == "auto" and len(legal_proposed) <= exact_enumeration_limit
    )
    if solver == "exact_enumeration" and len(legal_proposed) > exact_enumeration_limit:
        raise ValueError(
            "exact_enumeration solver requires proposal size "
            f"<= {exact_enumeration_limit}, got {len(legal_proposed)}"
        )

    if use_exact:
        projected = _exact_dependency_closed_projection(
            contract=contract,
            legal_proposed=legal_for_projection,
            forbidden=budgetable_forbidden,
            legal_residual=legal_residual,
            proposed_action=proposed_action,
            restoration_cost=restoration_cost,
            collision_budget=collision_budget,
            require_full_discharge=require_full_discharge,
            explicit_discharges=explicit_discharges,
        )
        projection_claim = "exact_dependency_closed_projection"
        readmission_rule = "exact_enumeration_small_graph"
    else:
        projected = _greedy_dependency_closed_projection(
            contract=contract,
            legal_proposed=legal_for_projection,
            forbidden=budgetable_forbidden,
            legal_residual=legal_residual,
            proposed_action=proposed_action,
            restoration_cost=restoration_cost,
            collision_budget=collision_budget,
            explicit_discharges=explicit_discharges,
        )
        projection_claim = "feasible_dependency_closed_projection"
        readmission_rule = "greedy_ascending_restoration_cost"

    # Closure may drop re-admitted colliding units; realize ``Lambda`` over what
    # actually survives so the reported collision matches the executed footprint.
    kept_colliding = projected & forbidden
    spent = restoration_weighted_collision(kept_colliding, restoration_cost)

    trimmed = proposed - projected

    # G(a') subset R(z): residual obligations whose full required footprint is covered.
    discharged = _discharged_obligations(
        contract,
        projected,
        explicit_discharges=explicit_discharges,
    )

    if legal_residual:
        progress_units = len(projected & legal_residual)
    else:
        progress_units = len(projected)
    if require_full_discharge:
        progress_units = len(discharged)

    if (not projected and not discharged) or (
        require_full_discharge and not discharged
    ):
        decision: Literal["project_bounded", "project_abstain", "project_identity"] = (
            "project_abstain"
        )
    elif not trimmed:
        decision = "project_identity"
    else:
        decision = "project_bounded"

    return ActionProjection(
        surface=contract.surface,
        action_id=proposed_action.action_id,
        proposed_footprint=proposed,
        projected_footprint=projected,
        trimmed_footprint=frozenset(trimmed),
        readmitted_footprint=frozenset(kept_colliding),
        lambda_collision=spent,
        collision_budget=collision_budget,
        progress_units=progress_units,
        discharged_obligation_count=len(discharged),
        discharged_obligations=discharged,
        projection_claim=projection_claim,
        readmission_rule=readmission_rule,
        decision=decision,
    )


def _obligation_key(obligation: Obligation) -> tuple[str, str]:
    return (obligation.status, obligation.obligation_id)


def _residual_write_footprint(contract: StateWriteContract) -> frozenset[str]:
    return frozenset(
        footprint for obligation in contract.residual for footprint in obligation.req()
    )


def _greedy_dependency_closed_projection(
    *,
    contract: StateWriteContract,
    legal_proposed: frozenset[str],
    forbidden: frozenset[str],
    legal_residual: frozenset[str],
    proposed_action: ProposedTargetAction,
    restoration_cost: Mapping[str, float] | None,
    collision_budget: float,
    explicit_discharges: tuple[str, ...],
) -> frozenset[str]:
    safe = legal_proposed - forbidden
    if contract.has_residual_state_change() and legal_residual:
        safe = safe & legal_residual

    colliding = sorted(
        legal_proposed & forbidden,
        key=lambda u: (
            0 if (not legal_residual or u in legal_residual) else 1,
            _unit_restoration_cost(u, restoration_cost),
            u,  # deterministic tie-break so re-admission is reproducible
        ),
    )
    readmitted: set[str] = set()
    spent = 0.0
    for unit in colliding:
        cost = _unit_restoration_cost(unit, restoration_cost)
        if spent + cost <= collision_budget:
            readmitted.add(unit)
            spent += cost

    projected = _close_under_dependencies(
        frozenset(safe | readmitted),
        proposed=legal_proposed,
        effect_groups=proposed_action.effect_groups,
        dependencies=proposed_action.dependency_map(),
    )
    return _prune_non_progress_colliders(
        projected,
        forbidden=forbidden,
        contract=contract,
        legal_residual=legal_residual,
        proposed=legal_proposed,
        effect_groups=proposed_action.effect_groups,
        dependencies=proposed_action.dependency_map(),
        explicit_discharges=explicit_discharges,
    )


def _exact_dependency_closed_projection(
    *,
    contract: StateWriteContract,
    legal_proposed: frozenset[str],
    forbidden: frozenset[str],
    legal_residual: frozenset[str],
    proposed_action: ProposedTargetAction,
    restoration_cost: Mapping[str, float] | None,
    collision_budget: float,
    require_full_discharge: bool,
    explicit_discharges: tuple[str, ...],
) -> frozenset[str]:
    candidate_units = _candidate_units_for_projection(
        contract=contract,
        legal_proposed=legal_proposed,
        forbidden=forbidden,
        legal_residual=legal_residual,
    )
    units = tuple(sorted(candidate_units))
    best: frozenset[str] = frozenset()
    best_score: tuple[int, int, float, int, tuple[str, ...]] | None = None

    dependency_map = proposed_action.dependency_map()
    for mask in range(1 << len(units)):
        candidate = frozenset(units[i] for i in range(len(units)) if mask & (1 << i))
        if (
            _close_under_dependencies(
                candidate,
                proposed=legal_proposed,
                effect_groups=proposed_action.effect_groups,
                dependencies=dependency_map,
            )
            != candidate
        ):
            continue
        collision = restoration_weighted_collision(
            candidate & forbidden, restoration_cost
        )
        if collision > collision_budget:
            continue
        discharged = _discharged_obligations(
            contract,
            candidate,
            explicit_discharges=explicit_discharges,
        )
        if legal_residual:
            progress_units = len(candidate & legal_residual)
        else:
            progress_units = len(candidate)
        if require_full_discharge and not discharged:
            continue
        score = (
            len(discharged),
            progress_units,
            len(candidate),
            -collision,
            tuple(sorted(candidate)),
        )
        if best_score is None or score > best_score:
            best = candidate
            best_score = score
    return best


def _candidate_units_for_projection(
    *,
    contract: StateWriteContract,
    legal_proposed: frozenset[str],
    forbidden: frozenset[str],
    legal_residual: frozenset[str],
) -> frozenset[str]:
    if contract.has_residual_state_change() and legal_residual:
        return (legal_proposed & legal_residual) | (legal_proposed & forbidden)
    return legal_proposed


def _progress_score(
    contract: StateWriteContract,
    projected: frozenset[str],
    legal_residual: frozenset[str],
    explicit_discharges: tuple[str, ...],
) -> tuple[int, int]:
    discharged = _discharged_obligations(
        contract,
        projected,
        explicit_discharges=explicit_discharges,
    )
    progress_units = (
        len(projected & legal_residual) if legal_residual else len(projected)
    )
    return (len(discharged), progress_units)


def _prune_non_progress_colliders(
    projected: frozenset[str],
    *,
    forbidden: frozenset[str],
    contract: StateWriteContract,
    legal_residual: frozenset[str],
    proposed: frozenset[str],
    effect_groups: tuple[frozenset[str], ...],
    dependencies: Mapping[str, frozenset[str]],
    explicit_discharges: tuple[str, ...],
) -> frozenset[str]:
    """Drop re-admitted colliders that do not preserve any residual progress."""

    kept = set(projected)
    changed = True
    while changed:
        changed = False
        current = frozenset(kept)
        current_score = _progress_score(
            contract,
            current,
            legal_residual,
            explicit_discharges,
        )
        for unit in sorted(current & forbidden):
            candidate = frozenset(kept - {unit})
            if (
                _close_under_dependencies(
                    candidate,
                    proposed=proposed,
                    effect_groups=effect_groups,
                    dependencies=dependencies,
                )
                != candidate
            ):
                continue
            if (
                _progress_score(
                    contract,
                    candidate,
                    legal_residual,
                    explicit_discharges,
                )
                == current_score
            ):
                kept.remove(unit)
                changed = True
                break
    return frozenset(kept)


def _unit_restoration_cost(
    unit: str,
    restoration_cost: Mapping[str, float] | None,
) -> float:
    if restoration_cost is None:
        return DEFAULT_RESTORATION_COST
    return restoration_cost.get(unit, DEFAULT_RESTORATION_COST)


def _discharged_obligations(
    contract: StateWriteContract,
    projected: frozenset[str],
    *,
    explicit_discharges: tuple[str, ...] = (),
) -> tuple[str, ...]:
    covered = {
        obligation.obligation_id
        for obligation in contract.residual
        if obligation.req() and obligation.req() <= projected
    }
    covered.update(explicit_discharges)
    return tuple(sorted(covered))


def _explicit_discharged_obligations(
    contract: StateWriteContract,
    proposed_action: ProposedTargetAction,
) -> tuple[str, ...]:
    if not proposed_action.terminal_completion:
        return ()
    residual_ids = {obligation.obligation_id for obligation in contract.residual}
    return tuple(
        sorted(proposed_action.discharged_obligation_ids & residual_ids)
    )


def _close_under_dependencies(
    projected: frozenset[str],
    *,
    proposed: frozenset[str],
    effect_groups: tuple[frozenset[str], ...],
    dependencies: Mapping[str, frozenset[str]],
) -> frozenset[str]:
    """Enforce dependency-closed sub-actions ``F(a,z)`` on a flat projection.

    Removes any unit that would leave the executable sub-action incoherent: a
    partially kept atomic effect group (some proposed units kept, some trimmed)
    collapses to fully trimmed, and a kept unit whose prerequisite units are not
    all kept is dropped. Iterates to a fixpoint over a monotonically shrinking
    set, so it terminates. A no-op when no groups/dependencies are declared, so
    the flat per-unit projection is recovered exactly.
    """

    if not effect_groups and not dependencies:
        return projected

    kept = set(projected)
    changed = True
    while changed:
        changed = False
        for group in effect_groups:
            group_proposed = frozenset(group) & proposed
            inside = group_proposed & kept
            if inside and (group_proposed - kept):
                kept -= inside
                changed = True
        for unit, prereqs in dependencies.items():
            if unit in kept and not (frozenset(prereqs) <= kept):
                kept.discard(unit)
                changed = True
    return frozenset(kept)

"""Shared CFRC compiler core.

This module is benchmark-independent. It compiles a verifier-issued CFRC
frontier and a typed proposal graph into a selected residual suffix, or abstains.
It does not parse dataset payloads and does not execute raw adapter tool calls.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from heapq import heapify, heappop, heappush
from typing import Any, Literal

from papc.certspec.cfrc_ir import (
    AuthorityLattice,
    EvidenceFact,
    ProposalGraph,
    ProposalNode,
)
from papc.certspec.executable_closure import (
    PROVENANCE_SELECTED_SUFFIX_NODE,
    ExecutableClosure,
)
from papc.certspec.minimum_authority_selector import (
    MinimumAuthoritySelection,
    SelectionCandidate,
    select_minimum_authority,
)
from papc.certspec.slot_candidate_generator import SlotConstraints, SlotEnumerationError
from papc.certspec.state_write_contract import ActionProjection, StateWriteContract

CFRCFrontierTiming = Literal["post_write", "pre_write"]
CFRCCompileMode = Literal["strict_core", "authority_lifted_extension"]
CFRCBindingKind = Literal["tool_call", "message", "none"]
CFRCCompileDecision = Literal["execute", "abstain", "block"]

AUTHORITY_LIFT_LABELS: frozenset[str] = frozenset(
    {
        "ell2",
        "ell2_temporal_or_irreversible_repair",
        "ell3",
        "ell3_bounded_local_redo",
    }
)
DEFAULT_REJECTED_AUTHORITIES: frozenset[str] = frozenset(
    {
        "ell_upper",
        "ell_upper_full_agent_or_unclassified",
    }
)
DEFAULT_STRICT_ALLOWED_AUTHORITIES: frozenset[str] = frozenset({"ell0", "ell1"})
_FORBIDDEN_PUBLIC_TOKENS: tuple[str, ...] = (
    "task_id",
    "gold",
    "direct_outcome",
    "scorer",
    "answer_sketch",
    "private_root",
    "private_receipt",
    "fixed_args",
    "fixed_query",
    "fixed_kb_query",
    "kb_query",
    "reward",
)


@dataclass(frozen=True)
class CFRCFrontier:
    """Verifier-issued frontier consumed by the shared compiler.

    ``allowed_effect_units`` is a restriction, not a permission list: an empty
    set means the frontier declares no unit restriction, so the compiler's
    ``selected_effect_outside_frontier_allowed_units`` check does not run. It
    does **not** mean "no effect is permitted". Airline and the ToolSandbox
    controller rely on the empty default while still selecting write nodes. A
    surface whose residual can be discharged without any write (a declaration
    path) therefore builds an empty frontier for a reason unrelated to
    permission, and must reject an effect-bearing node itself rather than
    expecting this check to do it.
    """

    contract: StateWriteContract
    allowed_effect_units: frozenset[str] = field(default_factory=frozenset)
    evidence_facts: tuple[EvidenceFact, ...] = ()
    frontier_timing: CFRCFrontierTiming = "post_write"
    pending_carrier_refs: tuple[str, ...] = ()
    executed_commitment_refs: tuple[str, ...] = ()
    restoration_cost: tuple[tuple[str, float], ...] | Mapping[str, float] = ()

    def __post_init__(self) -> None:
        if self.frontier_timing not in ("post_write", "pre_write"):
            raise ValueError("frontier_timing must be post_write or pre_write")
        _sorted_unique(self.allowed_effect_units, field_name="allowed_effect_units")
        _ordered_unique(self.pending_carrier_refs, field_name="pending_carrier_refs")
        _ordered_unique(
            self.executed_commitment_refs, field_name="executed_commitment_refs"
        )
        object.__setattr__(
            self, "restoration_cost", _normalize_restoration_cost(self.restoration_cost)
        )

    @property
    def verified_carrier_refs(self) -> frozenset[str]:
        if self.frontier_timing == "pre_write":
            return frozenset(self.pending_carrier_refs) | frozenset(
                self.executed_commitment_refs
            )
        return frozenset(self.executed_commitment_refs)

    @property
    def restoration_cost_map(self) -> dict[str, float]:
        return {unit: cost for unit, cost in self.restoration_cost}

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "papc.cfrc_compiler.frontier.v1",
            "contract_summary": {
                "surface": _validate_public_safe_text(
                    self.contract.surface,
                    field_name="contract_surface",
                ),
                "derivation_timing": self.contract.derivation_timing,
                "closed_obligation_count": len(self.contract.closed),
                "residual_obligation_count": len(self.contract.residual),
                "forbidden_write_count": len(self.contract.forbidden_writes),
                "state_change_required": self.contract.state_change_required,
                "user_facing_message_required": self.contract.user_facing_message_required,
            },
            "allowed_effect_units": sorted(self.allowed_effect_units),
            "evidence_fact_ids": [
                fact.fact_id
                for fact in sorted(self.evidence_facts, key=lambda item: item.fact_id)
            ],
            "frontier_timing": self.frontier_timing,
            "verified_carrier_policy": _verified_carrier_policy(self),
        }


@dataclass(frozen=True)
class CFRCCompileConfig:
    """Compiler policy for strict core or bounded authority-lifted extension.

    ``B_call`` belongs to the proposer-call ledger. The compiler budget below is
    only the maximum number of executable suffix tool calls selected after a
    typed proposal graph has already been issued.
    """

    mode: CFRCCompileMode = "strict_core"
    authority_lattice: AuthorityLattice = field(
        default_factory=lambda: AuthorityLattice.from_ordered_labels(("ell0", "ell1"))
    )
    strict_allowed_authorities: frozenset[str] | tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_STRICT_ALLOWED_AUTHORITIES
    )
    allowed_authorities: frozenset[str] | tuple[str, ...] = ()
    rejected_authorities: frozenset[str] | tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_REJECTED_AUTHORITIES
    )
    collision_budget: float = 0.0
    suffix_tool_call_budget: int | None = None
    b_call_budget: int | None = None
    require_whole_graph_selection: bool = False
    require_action_receipt_witness: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ("strict_core", "authority_lifted_extension"):
            raise ValueError("mode must be strict_core or authority_lifted_extension")
        strict_allowed = frozenset(
            _ordered_unique(
                self.strict_allowed_authorities, field_name="strict_allowed_authorities"
            )
        )
        allowed = frozenset(
            _ordered_unique(
                self.allowed_authorities
                or tuple(_lattice_labels(self.authority_lattice)),
                field_name="allowed_authorities",
            )
        )
        rejected = (
            frozenset(
                _ordered_unique(
                    self.rejected_authorities, field_name="rejected_authorities"
                )
            )
            | DEFAULT_REJECTED_AUTHORITIES
        )
        object.__setattr__(self, "strict_allowed_authorities", strict_allowed)
        object.__setattr__(self, "allowed_authorities", allowed)
        object.__setattr__(self, "rejected_authorities", rejected)
        if self.collision_budget < 0:
            raise ValueError("collision_budget must be non-negative")
        suffix_budget = (
            self.suffix_tool_call_budget
            if self.suffix_tool_call_budget is not None
            else self.b_call_budget
        )
        if suffix_budget is None:
            suffix_budget = 1
        suffix_budget = int(suffix_budget)
        if suffix_budget < 0:
            raise ValueError("suffix_tool_call_budget must be non-negative")
        object.__setattr__(self, "suffix_tool_call_budget", suffix_budget)
        # Backward-compatible constructor/read alias. It is not the method-level B_call.
        object.__setattr__(self, "b_call_budget", suffix_budget)
        if not isinstance(self.require_action_receipt_witness, bool):
            raise ValueError("require_action_receipt_witness must be boolean")

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "papc.cfrc_compiler.config.v1",
            "mode": self.mode,
            "authority_lattice": self.authority_lattice.to_json(),
            "strict_allowed_authorities": sorted(self.strict_allowed_authorities),
            "allowed_authorities": sorted(self.allowed_authorities),
            "rejected_authorities": sorted(self.rejected_authorities),
            "collision_budget": self.collision_budget,
            "suffix_tool_call_budget": self.suffix_tool_call_budget,
            "require_whole_graph_selection": self.require_whole_graph_selection,
            "require_action_receipt_witness": self.require_action_receipt_witness,
        }


@dataclass(frozen=True)
class CFRCNodeBinding:
    """Executable suffix binding for one typed proposal node."""

    node_id: str
    executable_closure: ExecutableClosure
    suffix_payload: Any | None = None
    binding_kind: CFRCBindingKind = "none"
    carrier_refs: tuple[str, ...] = ()
    evidence_output_refs: tuple[str, ...] = ()
    evidence_receipt_refs: tuple[str, ...] = ()
    replica_receipt_ref: str = ""
    replica_execution_succeeded: bool | None = None

    def __post_init__(self) -> None:
        _validate_public_safe_text(self.node_id, field_name="node_id")
        if self.binding_kind not in ("tool_call", "message", "none"):
            raise ValueError("binding_kind must be tool_call, message, or none")
        self.executable_closure.assert_valid()
        _ordered_unique(self.carrier_refs, field_name="carrier_refs")
        _ordered_unique(self.evidence_output_refs, field_name="evidence_output_refs")
        _ordered_unique(self.evidence_receipt_refs, field_name="evidence_receipt_refs")
        for receipt_ref in self.evidence_receipt_refs:
            if not _is_opaque_sha256_ref(receipt_ref):
                raise ValueError(
                    "evidence_receipt_refs must contain opaque sha256 digests"
                )
        if self.replica_receipt_ref:
            _validate_public_safe_text(
                self.replica_receipt_ref,
                field_name="replica_receipt_ref",
            )
            if not _is_opaque_sha256_ref(self.replica_receipt_ref):
                raise ValueError(
                    "replica_receipt_ref must contain an opaque sha256 digest"
                )
        if self.replica_execution_succeeded is not None and not isinstance(
            self.replica_execution_succeeded,
            bool,
        ):
            raise ValueError("replica_execution_succeeded must be boolean or None")
        _sorted_unique(
            self.executable_closure.semantic_effects, field_name="semantic_effects"
        )
        _sorted_unique(
            self.executable_closure.argument_carriers, field_name="argument_carriers"
        )
        _sorted_unique(
            self.executable_closure.executable_footprint,
            field_name="executable_footprint",
        )
        _validate_public_safe_mapping(
            self.executable_closure.carrier_provenance,
            key_field_name="carrier_provenance_key",
            value_field_name="carrier_provenance_value",
        )

    def public_summary(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "binding_kind": self.binding_kind,
            "payload_present": self.suffix_payload is not None,
            "payload_shape": _payload_shape(self.suffix_payload),
            "carrier_ref_count": len(self.carrier_refs),
            "evidence_output_ref_count": len(self.evidence_output_refs),
            "evidence_receipt_ref_count": len(self.evidence_receipt_refs),
            "replica_receipt_ref_present": bool(self.replica_receipt_ref),
            "replica_execution_succeeded": self.replica_execution_succeeded,
            "closure_fully_executable": self.executable_closure.fully_executable,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "papc.cfrc_compiler.node_binding.v1",
            **self.public_summary(),
            "evidence_output_refs": list(self.evidence_output_refs),
            "evidence_receipt_refs": list(self.evidence_receipt_refs),
            **(
                {"replica_receipt_ref": self.replica_receipt_ref}
                if self.replica_receipt_ref
                else {}
            ),
            "executable_closure": self.executable_closure.to_json(),
        }


@dataclass(frozen=True)
class CFRCPromptContract:
    """Public-safe prompt/proposal contract exposed to a residual proposer."""

    frontier_timing: CFRCFrontierTiming
    mode: CFRCCompileMode
    residual_obligation_count: int
    allowed_effect_units: tuple[str, ...]
    evidence_fact_ids: tuple[str, ...]
    authority_lattice: AuthorityLattice
    allowed_authorities: tuple[str, ...]
    rejected_authorities: tuple[str, ...]
    suffix_tool_call_budget: int
    require_action_receipt_witness: bool
    verified_carrier_policy: Mapping[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "papc.cfrc_compiler.prompt_contract.v1",
            "frontier_timing": self.frontier_timing,
            "mode": self.mode,
            "residual_obligation_count": self.residual_obligation_count,
            "allowed_effect_units": list(self.allowed_effect_units),
            "evidence_fact_ids": list(self.evidence_fact_ids),
            "authority_lattice": self.authority_lattice.to_json(),
            "allowed_authorities": list(self.allowed_authorities),
            "rejected_authorities": list(self.rejected_authorities),
            "residual_discharge_rule": "complete_R_z_discharge_required",
            "suffix_tool_call_budget": self.suffix_tool_call_budget,
            "require_action_receipt_witness": self.require_action_receipt_witness,
            "verified_carrier_policy": dict(self.verified_carrier_policy),
        }


@dataclass(frozen=True)
class CFRCCompileResult:
    """Compiled residual suffix decision."""

    decision: CFRCCompileDecision
    selected_node_ids: tuple[str, ...] = ()
    selected_authority_label: str = ""
    projection: ActionProjection | None = None
    suffix_payloads: tuple[Any, ...] = ()
    compile_errors: tuple[str, ...] = ()
    minimum_authority_trace: tuple[Mapping[str, Any], ...] = ()
    public_summary: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "papc.cfrc_compiler.result.v1",
            "decision": self.decision,
            "selected_node_ids": list(self.selected_node_ids),
            "selected_authority_label": self.selected_authority_label,
            "projection": _projection_public_summary(self.projection),
            "suffix_payloads": [
                _payload_shape(payload) for payload in self.suffix_payloads
            ],
            "compile_errors": list(self.compile_errors),
            "minimum_authority_trace": [
                dict(row) for row in self.minimum_authority_trace
            ],
            "public_summary": dict(self.public_summary),
        }


def prompt_contract(
    frontier: CFRCFrontier,
    config: CFRCCompileConfig,
) -> CFRCPromptContract:
    """Build the public-safe proposer contract for a fixed CFRC frontier."""

    active_allowed = _active_allowed_authorities(config)
    return CFRCPromptContract(
        frontier_timing=frontier.frontier_timing,
        mode=config.mode,
        residual_obligation_count=len(frontier.contract.residual),
        allowed_effect_units=tuple(sorted(frontier.allowed_effect_units)),
        evidence_fact_ids=tuple(
            fact.fact_id
            for fact in sorted(frontier.evidence_facts, key=lambda item: item.fact_id)
        ),
        authority_lattice=config.authority_lattice,
        allowed_authorities=tuple(sorted(active_allowed)),
        rejected_authorities=tuple(sorted(config.rejected_authorities)),
        suffix_tool_call_budget=config.suffix_tool_call_budget,
        require_action_receipt_witness=config.require_action_receipt_witness,
        verified_carrier_policy=_verified_carrier_policy(frontier),
    )


def compile_cfrc_suffix(
    *,
    frontier: CFRCFrontier,
    proposal_graph: ProposalGraph,
    bindings: Mapping[str, CFRCNodeBinding],
    config: CFRCCompileConfig,
    slot_constraints: SlotConstraints | None = None,
) -> CFRCCompileResult:
    """Compile a typed proposal graph into a residual suffix or abstention.

    ``slot_constraints`` is the optional typed branch scaffold (``one_of`` /
    ``requires`` / ``excludes``). When supplied, the minimum-authority selector
    enumerates only the bounded legal slot assignments instead of every
    dependency-closed subset; the projection, budget, binding-feasibility, and
    minimum-authority selection are otherwise identical, so omitting it reproduces
    the strict-selector behavior exactly.
    """

    pre_errors = _pre_selection_errors(
        frontier=frontier,
        proposal_graph=proposal_graph,
        bindings=bindings,
        config=config,
    )
    if pre_errors:
        return _blocked_result(
            compile_errors=pre_errors,
            proposal_graph=proposal_graph,
            config=config,
            reason="pre_selection_validation_failed",
        )

    try:
        selection = select_minimum_authority(
            contract=frontier.contract,
            proposal_graph=proposal_graph,
            authority_lattice=config.authority_lattice,
            collision_budget=config.collision_budget,
            restoration_cost=frontier.restoration_cost_map or None,
            lift_authorities=_lift_authorities(config),
            slot_constraints=slot_constraints,
        )
    except KeyError as exc:
        return _blocked_result(
            compile_errors=("authority_unknown",),
            proposal_graph=proposal_graph,
            config=config,
            reason=str(exc),
        )
    except SlotEnumerationError:
        return _blocked_result(
            compile_errors=("slot_candidate_enumeration_error",),
            proposal_graph=proposal_graph,
            config=config,
            reason="slot_candidate_enumeration_error",
        )

    if selection.decision != "select":
        return _abstain_result(
            selection=selection,
            proposal_graph=proposal_graph,
            config=config,
        )

    selection_candidates = selection.candidates
    if config.require_whole_graph_selection:
        proposed_ids = frozenset(node.node_id for node in proposal_graph.nodes)
        selection_candidates = tuple(
            candidate
            for candidate in selection_candidates
            if len(candidate.selected_node_ids) == len(proposed_ids)
            and frozenset(candidate.selected_node_ids) == proposed_ids
        )
        if not selection_candidates:
            return _blocked_result(
                compile_errors=("whole_graph_selection_required",),
                proposal_graph=proposal_graph,
                config=config,
                reason="partial_graph_selection_rejected",
            )
    budgeted_candidates = _candidates_within_suffix_tool_call_budget(
        selection_candidates,
        config,
    )
    if not budgeted_candidates:
        return _blocked_result(
            compile_errors=("suffix_tool_call_budget_exceeded",),
            proposal_graph=proposal_graph,
            config=config,
            selection=(
                None if config.require_whole_graph_selection else selection
            ),
            reason="no_complete_candidate_within_suffix_tool_call_budget",
        )
    selection, post_errors, candidate_rejection_reasons = _binding_feasible_selection(
        frontier=frontier,
        selection=selection,
        proposal_graph=proposal_graph,
        config=config,
        bindings=bindings,
        candidates=budgeted_candidates,
    )
    if post_errors:
        return _blocked_result(
            compile_errors=post_errors,
            proposal_graph=proposal_graph,
            config=config,
            selection=selection,
            reason="no_binding_feasible_candidate",
            candidate_rejection_reasons=candidate_rejection_reasons,
        )
    if config.require_whole_graph_selection:
        proposed_ids = tuple(node.node_id for node in proposal_graph.nodes)
        selected_ids = tuple(node.node_id for node in selection.selected_nodes)
        if len(selected_ids) != len(proposed_ids) or set(selected_ids) != set(
            proposed_ids
        ):
            return _blocked_result(
                compile_errors=("whole_graph_selection_required",),
                proposal_graph=proposal_graph,
                config=config,
                selection=selection,
                reason="partial_graph_selection_rejected",
                candidate_rejection_reasons=candidate_rejection_reasons,
            )

    ordered_selected_nodes, _ = _dependency_ordered_selected_nodes(
        selection.selected_nodes
    )
    selected_ids = tuple(node.node_id for node in ordered_selected_nodes)
    suffix_payloads = tuple(
        bindings[node_id].suffix_payload
        for node_id in selected_ids
        if bindings[node_id].suffix_payload is not None
    )
    public_summary = _public_summary(
        decision="execute",
        proposal_graph=proposal_graph,
        config=config,
        selected_node_ids=selected_ids,
        selected_authority_label=selection.selected_authority_label,
        suffix_payloads=suffix_payloads,
        compile_errors=(),
        reason=selection.reason,
        candidate_rejection_reasons=candidate_rejection_reasons,
    )
    return CFRCCompileResult(
        decision="execute",
        selected_node_ids=selected_ids,
        selected_authority_label=selection.selected_authority_label,
        projection=selection.projection,
        suffix_payloads=suffix_payloads,
        compile_errors=(),
        minimum_authority_trace=_minimum_authority_trace(selection),
        public_summary=public_summary,
    )


def _pre_selection_errors(
    *,
    frontier: CFRCFrontier,
    proposal_graph: ProposalGraph,
    bindings: Mapping[str, CFRCNodeBinding],
    config: CFRCCompileConfig,
) -> tuple[str, ...]:
    errors: list[str] = []
    active_allowed = _active_allowed_authorities(config)
    lattice_labels = set(_lattice_labels(config.authority_lattice))
    for node in proposal_graph.nodes:
        if node.node_role == "discharge_declaration":
            errors.append("declaration_cannot_discharge_obligation")
        authority = node.authority_label
        if authority in config.rejected_authorities:
            errors.append("authority_rejected")
        if authority not in lattice_labels:
            errors.append("authority_unknown")
        if authority not in active_allowed:
            errors.append("authority_not_allowed")
        if config.mode == "strict_core" and authority in AUTHORITY_LIFT_LABELS:
            errors.append("strict_core_rejects_authority_lift")
    if config.require_action_receipt_witness:
        errors.extend(
            _action_receipt_witness_errors(
                frontier=frontier,
                proposal_graph=proposal_graph,
                bindings=bindings,
            )
        )
    return _error_tuple(errors)


def _action_receipt_witness_errors(
    *,
    frontier: CFRCFrontier,
    proposal_graph: ProposalGraph,
    bindings: Mapping[str, CFRCNodeBinding],
) -> tuple[str, ...]:
    """Validate proposal-time coverage without advancing live obligation state."""

    errors: list[str] = []
    required_units = {
        obligation.obligation_id: obligation.req()
        for obligation in frontier.contract.residual
    }
    claimed_units = {obligation_id: set() for obligation_id in required_units}
    claimed_obligations: set[str] = set()
    seen_obligation_units: set[tuple[str, str]] = set()
    receipt_owners: dict[str, set[str]] = {}
    for node_id, binding in bindings.items():
        if binding.replica_receipt_ref:
            receipt_owners.setdefault(binding.replica_receipt_ref, set()).add(node_id)

    for node in proposal_graph.nodes:
        if node.discharged_obligation_ids:
            errors.append("strict_coverage_must_be_action_bound")
        for claim in node.coverage_claims:
            obligation_id = claim.obligation_id
            obligation_units = required_units.get(obligation_id)
            if obligation_units is None:
                errors.append("coverage_claim_unknown_obligation")
            else:
                claimed_obligations.add(obligation_id)
                for unit in claim.unit_tokens:
                    pair = (obligation_id, unit)
                    if pair in seen_obligation_units:
                        errors.append("coverage_claim_duplicate_obligation_unit")
                    seen_obligation_units.add(pair)
                    if unit not in obligation_units:
                        errors.append("coverage_claim_unit_outside_obligation")
                    else:
                        claimed_units[obligation_id].add(unit)

            binding = bindings.get(node.node_id)
            if binding is None or not binding.replica_receipt_ref:
                errors.append("coverage_claim_replica_receipt_missing")
                continue
            owners = receipt_owners.get(claim.replica_receipt_ref, set())
            if claim.replica_receipt_ref != binding.replica_receipt_ref:
                errors.append(
                    "coverage_claim_cross_action_receipt"
                    if owners and node.node_id not in owners
                    else "coverage_claim_replica_receipt_mismatch"
                )
                continue
            if owners != {node.node_id}:
                errors.append("coverage_claim_cross_action_receipt")
            if binding.replica_execution_succeeded is False:
                errors.append("coverage_claim_replica_receipt_failed")
            elif binding.replica_execution_succeeded is not True:
                errors.append("coverage_claim_replica_receipt_status_missing")

    for obligation_id, obligation_units in required_units.items():
        if obligation_id not in claimed_obligations:
            errors.append("coverage_claim_missing_obligation")
        elif obligation_units - claimed_units[obligation_id]:
            errors.append("coverage_claim_missing_required_unit")
    return _error_tuple(errors)


def _candidates_within_suffix_tool_call_budget(
    candidates: Iterable[SelectionCandidate],
    config: CFRCCompileConfig,
) -> tuple[SelectionCandidate, ...]:
    return tuple(
        candidate
        for candidate in candidates
        if candidate.tool_call_count <= int(config.suffix_tool_call_budget or 0)
    )


def _selection_from_candidate(
    *,
    selection: MinimumAuthoritySelection,
    proposal_graph: ProposalGraph,
    config: CFRCCompileConfig,
    candidate: SelectionCandidate,
    candidates: tuple[SelectionCandidate, ...],
    reason: str,
) -> MinimumAuthoritySelection:
    selected_nodes = _nodes_for_candidate(proposal_graph, candidate)
    selected_ranks = {
        node.authority_label: config.authority_lattice.rank_for(node.authority_label)
        for node in selected_nodes
    }
    selected_authority_label = max(
        selected_ranks,
        key=lambda label: (selected_ranks[label], label),
    )
    return MinimumAuthoritySelection(
        decision="select",
        selected_nodes=selected_nodes,
        selected_authority_label=selected_authority_label,
        selected_authority_rank=max(selected_ranks.values()),
        proposed_action=candidate.proposed_action,
        projection=candidate.projection,
        considered_candidate_count=selection.considered_candidate_count,
        candidates=candidates,
        reason=reason,
    )


def _binding_feasible_selection(
    *,
    frontier: CFRCFrontier,
    selection: MinimumAuthoritySelection,
    proposal_graph: ProposalGraph,
    config: CFRCCompileConfig,
    bindings: Mapping[str, CFRCNodeBinding],
    candidates: tuple[SelectionCandidate, ...],
) -> tuple[MinimumAuthoritySelection, tuple[str, ...], tuple[str, ...]]:
    first_rejected_selection: MinimumAuthoritySelection | None = None
    first_rejected_errors: tuple[str, ...] = ()
    rejected_reasons: list[str] = []
    for candidate in candidates:
        candidate_selection = _selection_from_candidate(
            selection=selection,
            proposal_graph=proposal_graph,
            config=config,
            candidate=candidate,
            candidates=candidates,
            reason=(
                "minimum_authority_dependency_closed_projection_"
                "within_suffix_tool_call_budget_and_binding"
            ),
        )
        errors = _post_selection_errors(
            frontier=frontier,
            selection=candidate_selection,
            bindings=bindings,
        )
        if not errors:
            return candidate_selection, (), _error_tuple(rejected_reasons)
        if first_rejected_selection is None:
            first_rejected_selection = candidate_selection
            first_rejected_errors = errors
        rejected_reasons.extend(errors)
    if first_rejected_selection is None:
        return selection, ("no_budgeted_candidate",), ()
    return (
        first_rejected_selection,
        first_rejected_errors,
        _error_tuple(rejected_reasons),
    )


def _nodes_for_candidate(
    proposal_graph: ProposalGraph,
    candidate: SelectionCandidate,
) -> tuple[ProposalNode, ...]:
    by_id = proposal_graph.node_by_id()
    return tuple(by_id[node_id] for node_id in candidate.selected_node_ids)


def _post_selection_errors(
    *,
    frontier: CFRCFrontier,
    selection: MinimumAuthoritySelection,
    bindings: Mapping[str, CFRCNodeBinding],
) -> tuple[str, ...]:
    errors: list[str] = []
    ordered_nodes, order_errors = _dependency_ordered_selected_nodes(
        selection.selected_nodes
    )
    if order_errors:
        return order_errors
    selected_by_id = {node.node_id: node for node in ordered_nodes}
    frontier_carriers = frontier.verified_carrier_refs
    available_carriers = set(frontier_carriers)
    for node in ordered_nodes:
        node_errors_before = len(errors)
        node_id = node.node_id
        binding = bindings.get(node_id)
        if binding is None:
            errors.append("selected_node_missing_binding")
            continue
        if binding.node_id != node_id:
            errors.append("binding_node_id_mismatch")
        if binding.suffix_payload is None:
            errors.append("selected_binding_missing_payload")
        if binding.binding_kind == "none":
            errors.append("selected_binding_has_no_suffix")
        expected_binding_kind = {
            "tool_call": "tool_call",
            "message": "message",
        }.get(node.kind)
        if (
            expected_binding_kind is not None
            and binding.binding_kind != expected_binding_kind
        ):
            errors.append("binding_kind_mismatch")
        if node.node_role == "terminal_communication":
            if binding.binding_kind != "message":
                errors.append("binding_kind_mismatch")
        if node.evidence_output_refs:
            if binding.binding_kind != "tool_call":
                errors.append("binding_kind_mismatch")
            if tuple(binding.evidence_output_refs) != tuple(node.evidence_output_refs):
                errors.append("evidence_output_not_bound")
            if not binding.evidence_receipt_refs:
                errors.append("evidence_receipt_not_bound")
            if tuple(binding.evidence_receipt_refs) != tuple(
                ref for ref in node.provenance_refs if _is_opaque_sha256_ref(ref)
            ):
                errors.append("evidence_receipt_lineage_mismatch")
        elif binding.evidence_output_refs or binding.evidence_receipt_refs:
            errors.append("non_evidence_binding_claims_evidence")
        binding_carriers = (
            frozenset(binding.carrier_refs)
            | binding.executable_closure.argument_carriers
        )
        if not _carriers_available_to_node(
            carriers=binding_carriers,
            carrier_provenance=binding.executable_closure.carrier_provenance,
            frontier_carriers=frontier_carriers,
            available_carriers=frozenset(available_carriers),
            selected_by_id=selected_by_id,
            consumer=node,
        ):
            errors.append("binding_carrier_not_verified")
        node_effects = (
            node.executable_closure.semantic_effects
            if node.executable_closure is not None
            else node.semantic_effect_set
        )
        if binding.executable_closure.semantic_effects != node_effects:
            errors.append("binding_closure_semantic_effect_mismatch")
        if not binding.executable_closure.fully_executable:
            errors.append("binding_closure_not_fully_executable")
        if (
            frontier.allowed_effect_units
            and not node_effects <= frontier.allowed_effect_units
        ):
            errors.append("selected_effect_outside_frontier_allowed_units")
        node_carriers = (
            node.executable_closure.argument_carriers
            if node.executable_closure is not None
            else frozenset()
        )
        node_carrier_provenance = (
            node.executable_closure.carrier_provenance
            if node.executable_closure is not None
            else {}
        )
        if not _carriers_available_to_node(
            carriers=node_carriers,
            carrier_provenance=node_carrier_provenance,
            frontier_carriers=frontier_carriers,
            available_carriers=frozenset(available_carriers),
            selected_by_id=selected_by_id,
            consumer=node,
        ):
            errors.append("node_carrier_not_verified")
        if len(errors) == node_errors_before:
            available_carriers.update(node_effects)
            available_carriers.update(node.evidence_output_refs)
    return _error_tuple(errors)


def _dependency_ordered_selected_nodes(
    nodes: tuple[ProposalNode, ...],
) -> tuple[tuple[ProposalNode, ...], tuple[str, ...]]:
    by_id = {node.node_id: node for node in nodes}
    if any(
        dependency_id not in by_id
        for node in nodes
        for dependency_id in node.dependency_ids
    ):
        return (), ("selected_dependency_missing",)

    dependents: dict[str, list[str]] = {node_id: [] for node_id in by_id}
    indegree: dict[str, int] = {}
    for node_id, node in by_id.items():
        dependencies = set(node.dependency_ids)
        indegree[node_id] = len(dependencies)
        for dependency_id in dependencies:
            dependents[dependency_id].append(node_id)
    # Preserve the compiler's lexicographic tie order without re-scanning the
    # full graph after each emitted node.
    ready = [node_id for node_id, count in indegree.items() if count == 0]
    heapify(ready)
    emitted: list[str] = []
    while ready:
        node_id = heappop(ready)
        emitted.append(node_id)
        for candidate_id in dependents[node_id]:
            indegree[candidate_id] -= 1
            if indegree[candidate_id] == 0:
                heappush(ready, candidate_id)

    if len(emitted) != len(nodes):
        return (), ("selected_dependency_cycle",)
    return tuple(by_id[node_id] for node_id in emitted), ()


def _carriers_available_to_node(
    *,
    carriers: frozenset[str],
    carrier_provenance: Mapping[str, str],
    frontier_carriers: frozenset[str],
    available_carriers: frozenset[str],
    selected_by_id: Mapping[str, ProposalNode],
    consumer: ProposalNode,
) -> bool:
    for carrier in carriers:
        if carrier in frontier_carriers:
            continue
        if carrier_provenance.get(carrier) != PROVENANCE_SELECTED_SUFFIX_NODE:
            return False
        if carrier not in available_carriers:
            return False
        if not _carrier_declares_selected_dependency(
            carrier=carrier,
            selected_by_id=selected_by_id,
            consumer=consumer,
        ):
            return False
    return True


def _carrier_declares_selected_dependency(
    *,
    carrier: str,
    selected_by_id: Mapping[str, ProposalNode],
    consumer: ProposalNode,
) -> bool:
    for dependency_id in consumer.dependency_ids:
        producer = selected_by_id.get(dependency_id)
        if producer is None:
            continue
        producer_effects = (
            producer.executable_closure.semantic_effects
            if producer.executable_closure is not None
            else producer.semantic_effect_set
        )
        if carrier in producer_effects or carrier in producer.evidence_output_refs:
            return True
    return False


def _blocked_result(
    *,
    compile_errors: Iterable[str],
    proposal_graph: ProposalGraph,
    config: CFRCCompileConfig,
    reason: str,
    selection: MinimumAuthoritySelection | None = None,
    candidate_rejection_reasons: Iterable[str] = (),
) -> CFRCCompileResult:
    errors = _error_tuple(compile_errors)
    selected_ids = (
        tuple(node.node_id for node in selection.selected_nodes)
        if selection is not None
        else ()
    )
    selected_authority_label = (
        selection.selected_authority_label if selection is not None else ""
    )
    public_summary = _public_summary(
        decision="block",
        proposal_graph=proposal_graph,
        config=config,
        selected_node_ids=selected_ids,
        selected_authority_label=selected_authority_label,
        suffix_payloads=(),
        compile_errors=errors,
        reason=reason,
        candidate_rejection_reasons=_error_tuple(candidate_rejection_reasons),
    )
    return CFRCCompileResult(
        decision="block",
        selected_node_ids=(),
        selected_authority_label="",
        projection=selection.projection if selection is not None else None,
        suffix_payloads=(),
        compile_errors=errors,
        minimum_authority_trace=_minimum_authority_trace(selection),
        public_summary=public_summary,
    )


def _abstain_result(
    *,
    selection: MinimumAuthoritySelection,
    proposal_graph: ProposalGraph,
    config: CFRCCompileConfig,
) -> CFRCCompileResult:
    public_summary = _public_summary(
        decision="abstain",
        proposal_graph=proposal_graph,
        config=config,
        selected_node_ids=(),
        selected_authority_label="",
        suffix_payloads=(),
        compile_errors=(),
        reason=selection.reason,
    )
    return CFRCCompileResult(
        decision="abstain",
        selected_node_ids=(),
        selected_authority_label="",
        projection=None,
        suffix_payloads=(),
        compile_errors=(),
        minimum_authority_trace=_minimum_authority_trace(selection),
        public_summary=public_summary,
    )


def _public_summary(
    *,
    decision: CFRCCompileDecision,
    proposal_graph: ProposalGraph,
    config: CFRCCompileConfig,
    selected_node_ids: tuple[str, ...],
    selected_authority_label: str,
    suffix_payloads: tuple[Any, ...],
    compile_errors: tuple[str, ...],
    reason: str,
    candidate_rejection_reasons: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "decision": decision,
        "mode": config.mode,
        "proposal_graph_node_count": len(proposal_graph.nodes),
        "selected_node_count": len(selected_node_ids),
        "selected_node_ids": list(selected_node_ids),
        "selected_authority_label": selected_authority_label,
        "suffix_payload_count": len(suffix_payloads),
        "suffix_payload_shapes": [
            _payload_shape(payload) for payload in suffix_payloads
        ],
        "suffix_tool_call_budget": config.suffix_tool_call_budget,
        "require_whole_graph_selection": config.require_whole_graph_selection,
        "require_action_receipt_witness": config.require_action_receipt_witness,
        "compile_error_count": len(compile_errors),
        "compile_errors": list(compile_errors),
        "candidate_rejection_count": len(candidate_rejection_reasons),
        "candidate_rejection_reasons": list(candidate_rejection_reasons),
        "reason": _validate_public_safe_text(reason, field_name="reason"),
    }


def _minimum_authority_trace(
    selection: MinimumAuthoritySelection | None,
) -> tuple[Mapping[str, Any], ...]:
    if selection is None:
        return ()
    return tuple(
        {
            "selected_node_ids": list(candidate.selected_node_ids),
            "max_authority_rank": candidate.max_authority_rank,
            "complete_residual_discharge": candidate.complete_residual_discharge,
            "discharged_obligation_count": candidate.discharged_obligation_count,
            "total_residual_obligation_count": candidate.total_residual_obligation_count,
            "tool_call_count": candidate.tool_call_count,
        }
        for candidate in selection.candidates
    )


def _verified_carrier_policy(frontier: CFRCFrontier) -> dict[str, Any]:
    carrier_sets = ["K_executed"]
    if frontier.frontier_timing == "pre_write":
        carrier_sets.insert(0, "K_pending")
    return {
        "policy": "verified_carrier_refs_only",
        "frontier_timing": frontier.frontier_timing,
        "carrier_sets": carrier_sets,
        "verified_carrier_ref_count": len(frontier.verified_carrier_refs),
        "pending_carrier_ref_count": len(frontier.pending_carrier_refs),
        "executed_commitment_ref_count": len(frontier.executed_commitment_refs),
    }


def _active_allowed_authorities(config: CFRCCompileConfig) -> frozenset[str]:
    if config.mode == "strict_core":
        return frozenset(config.strict_allowed_authorities) & frozenset(
            config.allowed_authorities
        )
    return frozenset(config.allowed_authorities)


def _lift_authorities(config: CFRCCompileConfig) -> frozenset[str]:
    """Authorities allowed to spend restoration budget for bounded local redo.

    Strict core never lifts. In the authority-lifted extension the lift tier is
    the configured allowed authorities recognized as repair tiers
    (``AUTHORITY_LIFT_LABELS``, e.g. ``ell2``/``ell3_bounded_local_redo``), minus
    the rejected full-agent tier. Strict ``ell0``/``ell1`` never spend budget, so
    the minimum-authority ordering still prefers a no-redo discharge.
    """

    if config.mode == "strict_core":
        return frozenset()
    return (frozenset(config.allowed_authorities) & AUTHORITY_LIFT_LABELS) - frozenset(
        config.rejected_authorities
    )


def _normalize_restoration_cost(
    raw: tuple[tuple[str, float], ...] | Mapping[str, float],
) -> tuple[tuple[str, float], ...]:
    items = raw.items() if isinstance(raw, Mapping) else raw
    pairs = tuple(sorted((str(unit), float(cost)) for unit, cost in items))
    for unit, _cost in pairs:
        _validate_public_safe_text(unit, field_name="restoration_cost_unit")
    return pairs


def _lattice_labels(lattice: AuthorityLattice) -> tuple[str, ...]:
    return tuple(level.label for level in lattice.levels)


def _payload_shape(payload: Any | None) -> dict[str, Any]:
    if payload is None:
        return {"type": "none"}
    if isinstance(payload, Mapping):
        return {
            "type": "mapping",
            "key_count": len(payload),
            "public_keys": [
                _validate_public_safe_text(str(key), field_name="suffix_payload_key")
                for key in sorted(payload)
                if _is_public_safe_text(str(key))
            ],
        }
    if isinstance(payload, (tuple, list)):
        return {"type": "sequence", "length": len(payload)}
    return {"type": type(payload).__name__}


def _projection_public_summary(
    projection: ActionProjection | None,
) -> dict[str, Any] | None:
    if projection is None:
        return None
    return {
        "surface": _validate_public_safe_text(
            projection.surface, field_name="projection_surface"
        ),
        "decision": projection.decision,
        "proposed_footprint_size": len(projection.proposed_footprint),
        "projected_footprint_size": len(projection.projected_footprint),
        "trimmed_footprint_size": len(projection.trimmed_footprint),
        "readmitted_footprint_size": len(projection.readmitted_footprint),
        "lambda_collision": projection.lambda_collision,
        "collision_budget": projection.collision_budget,
        "progress_units": projection.progress_units,
        "discharged_obligation_count": projection.discharged_obligation_count,
        "projection_claim": _validate_public_safe_text(
            projection.projection_claim,
            field_name="projection_claim",
        ),
        "readmission_rule": _validate_public_safe_text(
            projection.readmission_rule,
            field_name="readmission_rule",
        ),
        "makes_progress": projection.makes_progress,
        "trims": projection.trims,
    }


def _ordered_unique(values: Iterable[str], *, field_name: str) -> tuple[str, ...]:
    items: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = _validate_public_safe_text(str(raw), field_name=field_name)
        if value in seen:
            continue
        seen.add(value)
        items.append(value)
    return tuple(items)


def _sorted_unique(values: Iterable[str], *, field_name: str) -> tuple[str, ...]:
    return tuple(sorted(_ordered_unique(values, field_name=field_name)))


def _validate_public_safe_mapping(
    values: Mapping[str, str],
    *,
    key_field_name: str,
    value_field_name: str,
) -> dict[str, str]:
    sanitized: dict[str, str] = {}
    for key, value in values.items():
        safe_key = _validate_public_safe_text(str(key), field_name=key_field_name)
        safe_value = _validate_public_safe_text(str(value), field_name=value_field_name)
        sanitized[safe_key] = safe_value
    return sanitized


def _validate_public_safe_text(value: str, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    lowered = text.lower()
    token_text = lowered.replace("-", "_").replace(" ", "_")
    if text.startswith("/"):
        raise ValueError(f"{field_name} must not contain absolute paths")
    if "\\" in text:
        raise ValueError(f"{field_name} must not contain filesystem separators")
    if ".json" in lowered:
        raise ValueError(f"{field_name} must not contain artifact filenames")
    for token in _FORBIDDEN_PUBLIC_TOKENS:
        if token in token_text:
            raise ValueError(
                f"{field_name} contains forbidden runtime-visible token: {token}"
            )
    return text


def _is_public_safe_text(value: str) -> bool:
    try:
        _validate_public_safe_text(value, field_name="value")
    except ValueError:
        return False
    return True


def _error_tuple(errors: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(errors)))


def _is_opaque_sha256_ref(value: str) -> bool:
    prefix = "sha256:"
    text = str(value)
    if not text.startswith(prefix):
        return False
    digest = text[len(prefix) :]
    return len(digest) == 64 and all(
        character in "0123456789abcdef" for character in digest
    )


__all__ = [
    "CFRCCompileConfig",
    "CFRCCompileResult",
    "CFRCFrontier",
    "CFRCNodeBinding",
    "CFRCPromptContract",
    "compile_cfrc_suffix",
    "prompt_contract",
]

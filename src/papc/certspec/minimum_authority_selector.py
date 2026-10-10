"""Deterministic minimum-authority selector over the shared CFRC IR."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from papc.certspec.cfrc_ir import AuthorityLattice, ProposalGraph, ProposalNode
from papc.certspec.slot_candidate_generator import (
    SlotConstraints,
    assignment_respects_slot_constraints,
    enumerate_slot_assignments,
)
from papc.certspec.state_write_contract import (
    ActionProjection,
    ProposedTargetAction,
    StateWriteContract,
    project_admissible_action,
)

SelectionDecision = Literal["select", "abstain"]


@dataclass(frozen=True)
class SelectionCandidate:
    """One dependency-closed candidate subgraph considered by the selector."""

    candidate_node_ids: tuple[str, ...]
    selected_node_ids: tuple[str, ...]
    max_authority_rank: int
    complete_residual_discharge: bool
    discharged_obligation_count: int
    total_residual_obligation_count: int
    write_scope_size: int
    selected_node_count: int
    tool_call_count: int
    proposed_action: ProposedTargetAction
    projection: ActionProjection

    def sort_key(self) -> tuple[int, int, int, int, int, int, tuple[str, ...]]:
        return (
            self.max_authority_rank,
            0 if self.complete_residual_discharge else 1,
            -self.discharged_obligation_count,
            self.write_scope_size,
            self.selected_node_count,
            self.tool_call_count,
            self.selected_node_ids,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "candidate_node_ids": list(self.candidate_node_ids),
            "selected_node_ids": list(self.selected_node_ids),
            "max_authority_rank": self.max_authority_rank,
            "complete_residual_discharge": self.complete_residual_discharge,
            "discharged_obligation_count": self.discharged_obligation_count,
            "total_residual_obligation_count": self.total_residual_obligation_count,
            "write_scope_size": self.write_scope_size,
            "selected_node_count": self.selected_node_count,
            "tool_call_count": self.tool_call_count,
            "proposed_action": self.proposed_action.to_json(),
            "projection": self.projection.to_json(),
        }


@dataclass(frozen=True)
class MinimumAuthoritySelection:
    """Selector result with public-safe diagnostics."""

    decision: SelectionDecision
    selected_nodes: tuple[ProposalNode, ...]
    selected_authority_label: str
    selected_authority_rank: int | None
    proposed_action: ProposedTargetAction | None
    projection: ActionProjection | None
    considered_candidate_count: int
    candidates: tuple[SelectionCandidate, ...]
    reason: str

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "papc.minimum_authority_selection.v1",
            "decision": self.decision,
            "selected_node_ids": [node.node_id for node in self.selected_nodes],
            "selected_authority_label": self.selected_authority_label,
            "selected_authority_rank": self.selected_authority_rank,
            "proposed_action": (
                self.proposed_action.to_json()
                if self.proposed_action is not None
                else None
            ),
            "projection": self.projection.to_json()
            if self.projection is not None
            else None,
            "considered_candidate_count": self.considered_candidate_count,
            "reason": self.reason,
            "candidates": [candidate.to_json() for candidate in self.candidates],
        }


def select_minimum_authority(
    *,
    contract: StateWriteContract,
    proposal_graph: ProposalGraph,
    authority_lattice: AuthorityLattice,
    collision_budget: float = 0.0,
    restoration_cost: Mapping[str, float] | None = None,
    lift_authorities: frozenset[str] | tuple[str, ...] = (),
    slot_constraints: SlotConstraints | None = None,
) -> MinimumAuthoritySelection:
    """Select the deterministic minimum-authority executable subgraph.

    ``collision_budget`` and ``restoration_cost`` enable bounded local redo: a
    candidate that contains a lift-authority node (``lift_authorities``, e.g. the
    pre-registered ``ell2`` repair tier) may re-admit already-committed units
    whose cumulative restoration cost ``Lambda`` stays within the budget. Strict
    candidates, and every caller that omits these arguments, keep ``budget = 0``
    and therefore the exact pre-redo behavior; units typed as irreversible
    (infinite restoration cost) are never re-admitted regardless of budget.

    ``slot_constraints`` enables the authority-lifted *branch* path: when a typed
    scaffold (``one_of`` / ``requires`` / ``excludes``) is supplied, the candidate
    subgraphs come from :func:`enumerate_slot_assignments` (bounded by the product
    of slot-group sizes) instead of the strict ``2**n`` dependency-closed subset
    enumeration. The projection, scoring, and minimum-authority tie-break below are
    identical for both paths, so ``slot_constraints=None`` reproduces the strict
    selector exactly.
    """

    nodes = tuple(
        node
        for node in sorted(proposal_graph.nodes, key=lambda item: item.node_id)
        if node.kind != "abstain"
        and node.executable
        and (
            node.semantic_effects
            or _node_discharged_obligation_ids(node)
            or node.node_role in {"evidence_acquisition", "terminal_communication"}
        )
    )
    if not nodes or not any(
        _node_semantic_effects(node) or _node_discharged_obligation_ids(node)
        for node in nodes
    ):
        return MinimumAuthoritySelection(
            decision="abstain",
            selected_nodes=(),
            selected_authority_label="",
            selected_authority_rank=None,
            proposed_action=None,
            projection=None,
            considered_candidate_count=0,
            candidates=(),
            reason="no_executable_cfrc_nodes",
        )

    by_id = {node.node_id: node for node in nodes}
    lift_labels = frozenset(lift_authorities)
    total_residual_obligation_count = len(contract.residual)
    residual_obligation_ids = frozenset(
        obligation.obligation_id for obligation in contract.residual
    )
    candidates: list[SelectionCandidate] = []
    seen_subsets: set[tuple[str, ...]] = set()
    for raw_subset_ids in _candidate_subsets(nodes, slot_constraints):
        subset_ids = _attach_required_terminal_communication(
            subset_ids=raw_subset_ids,
            nodes=nodes,
            required=contract.user_facing_message_required,
        )
        if subset_ids is None or subset_ids in seen_subsets:
            continue
        if slot_constraints is not None and not assignment_respects_slot_constraints(
            assignment=subset_ids,
            node_ids=tuple(node.node_id for node in nodes),
            dependency_map={node.node_id: node.dependency_ids for node in nodes},
            constraints=slot_constraints,
        ):
            continue
        seen_subsets.add(subset_ids)
        subset_nodes = tuple(by_id[node_id] for node_id in subset_ids)
        if contract.user_facing_message_required and not _satisfies_user_facing_message(
            subset_nodes, graph_nodes=nodes
        ):
            continue
        proposed_action = _build_proposed_action(
            contract=contract, nodes=subset_nodes, graph_nodes=nodes
        )
        if not proposed_action.discharged_obligation_ids <= residual_obligation_ids:
            continue
        subset_budget = (
            collision_budget
            if collision_budget > 0.0 and _subset_can_lift(subset_nodes, lift_labels)
            else 0.0
        )
        projection = project_admissible_action(
            contract,
            proposed_action,
            restoration_cost=restoration_cost,
            collision_budget=subset_budget,
            require_full_discharge=total_residual_obligation_count > 0,
        )
        if not projection.makes_progress:
            continue
        selected_nodes = _selected_nodes_from_projection(
            subset_nodes,
            projection,
            require_user_facing_message=contract.user_facing_message_required,
            graph_nodes=nodes,
        )
        if not selected_nodes:
            continue
        if contract.user_facing_message_required and not _satisfies_user_facing_message(
            selected_nodes, graph_nodes=nodes
        ):
            continue
        if not _terminal_dependencies_cover_selected_effects(selected_nodes):
            continue
        max_rank = max(
            authority_lattice.rank_for(node.authority_label) for node in selected_nodes
        )
        discharged_obligation_count = projection.discharged_obligation_count
        complete_residual_discharge = (
            total_residual_obligation_count == 0
            or discharged_obligation_count == total_residual_obligation_count
        )
        if total_residual_obligation_count > 0 and not complete_residual_discharge:
            continue
        candidates.append(
            SelectionCandidate(
                candidate_node_ids=subset_ids,
                selected_node_ids=tuple(node.node_id for node in selected_nodes),
                max_authority_rank=max_rank,
                complete_residual_discharge=complete_residual_discharge,
                discharged_obligation_count=discharged_obligation_count,
                total_residual_obligation_count=total_residual_obligation_count,
                write_scope_size=len(projection.projected_footprint),
                selected_node_count=len(selected_nodes),
                tool_call_count=sum(
                    node.effective_tool_call_count for node in selected_nodes
                ),
                proposed_action=proposed_action,
                projection=projection,
            )
        )

    if not candidates:
        return MinimumAuthoritySelection(
            decision="abstain",
            selected_nodes=(),
            selected_authority_label="",
            selected_authority_rank=None,
            proposed_action=None,
            projection=None,
            considered_candidate_count=0,
            candidates=(),
            reason="no_progressing_dependency_closed_candidate",
        )

    ordered_candidates = tuple(sorted(candidates, key=lambda item: item.sort_key()))
    best = ordered_candidates[0]
    selected_nodes = tuple(by_id[node_id] for node_id in best.selected_node_ids)
    selected_ranks = {
        node.authority_label: authority_lattice.rank_for(node.authority_label)
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
        proposed_action=best.proposed_action,
        projection=best.projection,
        considered_candidate_count=len(ordered_candidates),
        candidates=ordered_candidates,
        reason="minimum_authority_dependency_closed_projection",
    )


def _satisfies_user_facing_message(
    candidate: tuple[ProposalNode, ...] | set[str],
    *,
    graph_nodes: tuple[ProposalNode, ...],
    by_id: Mapping[str, ProposalNode] | None = None,
) -> bool:
    """Whether a candidate carries the user-facing message the contract requires.

    A graph may contain message nodes that are not the reply to the user: an
    explicit discharge declaration is a zero-authority binding carrier, not a
    turn. When the graph declares a terminal communication, only that node can
    satisfy the requirement, so a candidate cannot drop the reply and still look
    complete because it happens to carry a declaration.
    """

    if isinstance(candidate, (set, frozenset)):
        if by_id is None:
            raise ValueError("resolving a candidate of node ids requires by_id")
        nodes: tuple[ProposalNode, ...] = tuple(
            by_id[node_id] for node_id in candidate if node_id in by_id
        )
    else:
        nodes = tuple(candidate)
    if any(node.node_role == "terminal_communication" for node in graph_nodes):
        return any(node.node_role == "terminal_communication" for node in nodes)
    return any(node.kind == "message" for node in nodes)


def _build_proposed_action(
    *,
    contract: StateWriteContract,
    nodes: tuple[ProposalNode, ...],
    graph_nodes: tuple[ProposalNode, ...] = (),
) -> ProposedTargetAction:
    """Lift a candidate subgraph into the action the projection reasons about.

    ``terminal_completion`` gates whether explicit discharges count, so it uses
    the same terminal-aware rule as the user-facing message check: once a graph
    declares a reply, a candidate that drops it must not still collect the
    obligations a declaration states. A graph with no declared reply keeps the
    plain any-message rule, which is what every pre-declaration surface built.
    """

    write_footprint = frozenset(
        unit for node in nodes for unit in _node_semantic_effects(node)
    )
    dependencies: list[tuple[str, frozenset[str]]] = []
    node_by_id = {node.node_id: node for node in nodes}
    for node in nodes:
        prereq_units = frozenset(
            unit
            for dep_id in node.dependency_ids
            if dep_id in node_by_id
            for unit in _node_semantic_effects(node_by_id[dep_id])
        )
        if not prereq_units:
            continue
        for unit in _node_semantic_effects(node):
            dependencies.append((unit, prereq_units))
    return ProposedTargetAction(
        surface=contract.surface,
        action_id="+".join(node.node_id for node in nodes),
        write_footprint=write_footprint,
        writable_scope=write_footprint,
        effect_groups=tuple(
            effects for node in nodes if (effects := _node_semantic_effects(node))
        ),
        dependencies=tuple(dependencies),
        terminal_completion=_satisfies_user_facing_message(
            nodes, graph_nodes=graph_nodes or nodes
        ),
        discharged_obligation_ids=frozenset(
            obligation_id
            for node in nodes
            if node.node_role != "discharge_declaration"
            for obligation_id in _node_discharged_obligation_ids(node)
        ),
    )


def _candidate_subsets(
    nodes: tuple[ProposalNode, ...],
    slot_constraints: SlotConstraints | None,
) -> Iterable[tuple[str, ...]]:
    """Pick the candidate-subgraph source: bounded slot path or strict subsets.

    When a typed branch scaffold is supplied, candidate subgraphs are the bounded
    legal slot assignments. This includes constraints objects that only carry
    ``requires`` / ``excludes`` filters. Only ``slot_constraints=None`` keeps the
    strict dependency-closed subset enumeration unchanged.
    """

    if slot_constraints is not None:
        return enumerate_slot_assignments(
            node_ids=tuple(node.node_id for node in nodes),
            dependency_map={
                node.node_id: frozenset(node.dependency_ids) for node in nodes
            },
            constraints=slot_constraints,
        )
    return _dependency_closed_subsets(nodes)


# Exact 2**n subset enumeration stays byte-identical up to this width; wider
# proposal graphs (a one-response terminal plan can carry a read storm with a
# handful of writes) switch to effect-anchored enumeration below.
_EXACT_SUBSET_ENUMERATION_WIDTH = 16


def _dependency_closed_subsets(
    nodes: tuple[ProposalNode, ...],
) -> Iterable[tuple[str, ...]]:
    if len(nodes) <= _EXACT_SUBSET_ENUMERATION_WIDTH:
        yield from _exact_dependency_closed_subsets(nodes)
        return
    yield from _effect_anchored_dependency_closed_subsets(nodes)


def _exact_dependency_closed_subsets(
    nodes: tuple[ProposalNode, ...],
) -> Iterable[tuple[str, ...]]:
    node_ids = [node.node_id for node in nodes]
    node_bits = {node_id: 1 << index for index, node_id in enumerate(node_ids)}
    missing_bit = 1 << len(node_ids)
    dependency_masks: list[int] = []
    constrained_nodes = 0
    for index, node in enumerate(nodes):
        dependencies = 0
        for dependency_id in node.dependency_ids:
            dependencies |= node_bits.get(dependency_id, missing_bit)
        dependency_masks.append(dependencies)
        if dependencies:
            constrained_nodes |= 1 << index
    for mask in range(1, 1 << len(node_ids)):
        remaining = mask & constrained_nodes
        while remaining:
            bit = remaining & -remaining
            dependencies = dependency_masks[bit.bit_length() - 1]
            if dependencies & mask != dependencies:
                break
            remaining ^= bit
        else:
            # Materialize only closed subsets, in the original mask order.
            yield tuple(
                node_ids[index]
                for index in range(len(node_ids))
                if mask & (1 << index)
            )


def _effect_anchored_dependency_closed_subsets(
    nodes: tuple[ProposalNode, ...],
) -> Iterable[tuple[str, ...]]:
    """Dependency-closed candidates anchored on effect-bearing nodes.

    Choice ranges only over nodes that carry semantic effects, discharge
    obligations, or terminal communication; pure evidence nodes enter a
    candidate solely through the transitive dependency closure of the chosen
    anchors. Every yielded subset is dependency closed, so projection,
    scoring, and the minimum-authority ordering are unchanged. Candidates
    that differ from a yielded one only by disconnected evidence nodes are
    not generated: they project the same footprint and can never beat their
    closed core under the ordering. If even the anchor count exceeds the
    exact width, the single full-anchor closure is the only candidate.
    """

    node_ids = [node.node_id for node in nodes]
    order = {node_id: index for index, node_id in enumerate(node_ids)}
    dependency_map = {node.node_id: set(node.dependency_ids) for node in nodes}
    anchors = [
        node.node_id
        for node in nodes
        if _node_semantic_effects(node)
        or _node_discharged_obligation_ids(node)
        or node.kind == "message"
        or node.node_role == "terminal_communication"
    ]
    if not anchors:
        return
    if len(anchors) > _EXACT_SUBSET_ENUMERATION_WIDTH:
        anchor_subsets: Iterable[tuple[str, ...]] = (tuple(anchors),)
    else:
        anchor_subsets = (
            tuple(
                anchors[index]
                for index in range(len(anchors))
                if mask & (1 << index)
            )
            for mask in range(1, 1 << len(anchors))
        )
    seen: set[tuple[str, ...]] = set()
    for anchor_subset in anchor_subsets:
        closed = set(anchor_subset)
        frontier = list(anchor_subset)
        dependency_missing = False
        while frontier and not dependency_missing:
            current = frontier.pop()
            for dependency_id in dependency_map.get(current, ()):
                if dependency_id not in dependency_map:
                    # Mirrors the exact enumerator: a node depending on
                    # something outside the executable graph is never
                    # part of a closed subset.
                    dependency_missing = True
                    break
                if dependency_id not in closed:
                    closed.add(dependency_id)
                    frontier.append(dependency_id)
        if dependency_missing:
            continue
        subset = tuple(sorted(closed, key=order.__getitem__))
        if subset in seen:
            continue
        seen.add(subset)
        yield subset


def _subset_can_lift(
    subset_nodes: tuple[ProposalNode, ...],
    lift_authorities: frozenset[str],
) -> bool:
    """Whether a candidate may spend restoration budget for bounded local redo.

    Only nodes whose authority label is in the configured lift tier (e.g.
    ``ell2``) may re-admit committed units. Strict ``ell0``/``ell1`` candidates
    never spend budget, so the minimum-authority ordering still prefers a
    no-redo discharge whenever one fully closes the residual obligations.
    """

    return any(node.authority_label in lift_authorities for node in subset_nodes)


def _node_semantic_effects(node: ProposalNode) -> frozenset[str]:
    if node.executable_closure is not None:
        return node.executable_closure.semantic_effects
    return frozenset(node.semantic_effects)


def _node_discharged_obligation_ids(node: ProposalNode) -> frozenset[str]:
    return frozenset(node.discharged_obligation_ids) | frozenset(
        claim.obligation_id for claim in node.coverage_claims
    )


def _selected_nodes_from_projection(
    nodes: tuple[ProposalNode, ...],
    projection: ActionProjection,
    *,
    require_user_facing_message: bool,
    graph_nodes: tuple[ProposalNode, ...] = (),
) -> tuple[ProposalNode, ...]:
    projected = projection.projected_footprint
    by_id = {node.node_id: node for node in nodes}
    selected_ids = {
        node.node_id
        for node in nodes
        if (
            _node_semantic_effects(node)
            and _node_semantic_effects(node) <= projected
        )
        or (
            _node_discharged_obligation_ids(node)
            and _node_discharged_obligation_ids(node)
            <= frozenset(projection.discharged_obligations)
        )
    }
    _close_selected_dependencies(selected_ids=selected_ids, by_id=by_id)
    if require_user_facing_message and not _satisfies_user_facing_message(
        selected_ids, graph_nodes=graph_nodes or nodes, by_id=by_id
    ):
        terminal = _terminal_with_admissible_evidence_tail(
            nodes=nodes,
            selected_ids=selected_ids,
            by_id=by_id,
        )
        if terminal is not None:
            selected_ids.add(terminal.node_id)
            _close_selected_dependencies(selected_ids=selected_ids, by_id=by_id)
    return tuple(
        sorted(
            (by_id[node_id] for node_id in selected_ids),
            key=lambda item: item.node_id,
        )
    )


def _terminal_with_admissible_evidence_tail(
    *,
    nodes: tuple[ProposalNode, ...],
    selected_ids: set[str],
    by_id: Mapping[str, ProposalNode],
) -> ProposalNode | None:
    """Attach a terminal whose unselected tail contains evidence only.

    A valid residual may execute a write, perform readback or unrelated support
    reads, and only then publish its terminal response. Projection selects the
    effect node first. The former direct-dependency check could not attach the
    terminal because its pure evidence tail had not itself been selected, so a
    write -> readback -> terminal graph incorrectly had no progressing candidate.

    We may add the dependency closure only when every effect-bearing or explicit
    discharge node in that closure is already selected. This admits evidence
    acquisition after a selected effect but never smuggles in an unprojected
    write.
    """

    for terminal in nodes:
        if terminal.node_role != "terminal_communication":
            continue
        closure = _dependency_closure(terminal.node_id, by_id=by_id)
        if closure is None:
            continue
        unselected = closure - selected_ids - {terminal.node_id}
        if any(
            _node_semantic_effects(by_id[node_id])
            or _node_discharged_obligation_ids(by_id[node_id])
            for node_id in unselected
        ):
            continue
        selected_ids.update(unselected)
        return terminal
    return None


def _dependency_closure(
    node_id: str,
    *,
    by_id: Mapping[str, ProposalNode],
) -> set[str] | None:
    if node_id not in by_id:
        return None
    closure = {node_id}
    frontier = [node_id]
    while frontier:
        current = frontier.pop()
        for dependency_id in by_id[current].dependency_ids:
            if dependency_id not in by_id:
                return None
            if dependency_id not in closure:
                closure.add(dependency_id)
                frontier.append(dependency_id)
    return closure


def _close_selected_dependencies(
    *,
    selected_ids: set[str],
    by_id: Mapping[str, ProposalNode],
) -> None:
    stack = list(selected_ids)
    while stack:
        node = by_id[stack.pop()]
        for dependency_id in node.dependency_ids:
            if dependency_id not in by_id or dependency_id in selected_ids:
                continue
            selected_ids.add(dependency_id)
            stack.append(dependency_id)


def _attach_required_terminal_communication(
    *,
    subset_ids: tuple[str, ...],
    nodes: tuple[ProposalNode, ...],
    required: bool,
) -> tuple[str, ...] | None:
    """Attach the unique zero-effect terminal after its dependencies are present.

    Slot enumeration intentionally only pulls prerequisites backward.  A terminal
    communication is a dependent, so it is attached here only when the contract
    requires a user-facing message and its complete dependency set is already in
    the candidate.  It never makes an otherwise-incomplete effect projection
    complete.
    """

    subset = set(subset_ids)
    if not required or _satisfies_user_facing_message(
        subset, graph_nodes=nodes, by_id={node.node_id: node for node in nodes}
    ):
        return tuple(sorted(subset))
    terminal = next(
        (
            node
            for node in nodes
            if node.node_role == "terminal_communication"
            and set(node.dependency_ids) <= subset
        ),
        None,
    )
    if terminal is None:
        return None
    subset.add(terminal.node_id)
    return tuple(sorted(subset))


def _terminal_dependencies_cover_selected_effects(
    nodes: tuple[ProposalNode, ...],
) -> bool:
    terminal = next(
        (node for node in nodes if node.node_role == "terminal_communication"),
        None,
    )
    if terminal is None:
        return True
    by_id = {node.node_id: node for node in nodes}
    ancestors: set[str] = set()
    stack = list(terminal.dependency_ids)
    while stack:
        node_id = stack.pop()
        if node_id in ancestors or node_id not in by_id:
            continue
        ancestors.add(node_id)
        stack.extend(by_id[node_id].dependency_ids)
    selected_effect_ids = {
        node.node_id
        for node in nodes
        if node.node_role == "obligation_effect" and bool(_node_semantic_effects(node))
    }
    return selected_effect_ids <= ancestors


__all__ = [
    "MinimumAuthoritySelection",
    "SelectionCandidate",
    "select_minimum_authority",
]

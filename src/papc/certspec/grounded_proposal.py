"""Receipt-grounded proposal bridge shared by CFRC surface adapters.

The strong model acts through native tools and observes their receipts.
Surface code records those realized calls as ``GroundedToolCall`` objects and
supplies only the surface-specific effect and obligation projection. This
module then performs the mechanical lift into the shared ``ProposalGraph`` IR.

Concrete arguments, results, and replay payloads remain private.  The public IR
contains only stable node ids, opaque receipt digests, mechanically derived
dependencies, and surface-certified abstract effects.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from papc.certspec.cfrc_compiler import CFRCNodeBinding
from papc.certspec.cfrc_ir import ProposalCoverageClaim, ProposalGraph, ProposalNode
from papc.certspec.executable_closure import build_executable_closure
from papc.certspec.receipt_provenance import (
    ValueProvenance,
    carrier_unit_for,
    derive_value_provenance,
    provenance_carrier_tags,
    provenance_carrier_units,
    provenance_public_summary,
)

SCHEMA = "papc.grounded_proposal_bridge.v1"
TERMINAL_NODE_ID = "grounded_terminal"
DECLARATION_NODE_PREFIX = "grounded_declaration_"
DischargeMode = Literal["decline", "clarify"]


@dataclass(frozen=True)
class GroundedToolCall:
    """One native call and its observed receipt."""

    node_id: str
    tool_name: str
    arguments: Mapping[str, Any]
    result: Any
    proposal_round: int
    succeeded: bool = True
    visible_receipt_refs: tuple[str, ...] = ()
    replay_payload: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not str(self.node_id).strip():
            raise ValueError("grounded call node_id must be non-empty")
        if not str(self.tool_name).strip():
            raise ValueError("grounded call tool_name must be non-empty")
        if int(self.proposal_round) <= 0:
            raise ValueError("grounded call proposal_round must be positive")
        if len(self.visible_receipt_refs) != len(set(self.visible_receipt_refs)):
            raise ValueError("grounded call visible receipts must be unique")
        _canonical_json_bytes(self.arguments)
        _canonical_json_bytes(self.result)

    @property
    def receipt_ref(self) -> str:
        return receipt_ref(
            tool_name=self.tool_name,
            arguments=self.arguments,
            result=self.result,
            instance_id=self.node_id,
        )


@dataclass(frozen=True)
class GroundedDischargeDeclaration:
    """One zero-authority request for a contract-scoped failure return.

    A declaration records that the model declines an open obligation or needs
    clarification. It carries no semantic effect, contributes no successful
    obligation coverage, and is rejected by the compiler as a completion. Keeping
    it in the proposal trace makes the failure intent auditable without confusing
    it with an action--receipt witness.
    """

    obligation_id: str
    mode: DischargeMode
    basis: str

    def __post_init__(self) -> None:
        if not str(self.obligation_id).strip():
            raise ValueError("discharge declaration requires an obligation id")
        if self.mode not in ("decline", "clarify"):
            raise ValueError("discharge declaration mode must be decline or clarify")
        if not str(self.basis).strip():
            raise ValueError("discharge declaration requires a non-empty basis")


@dataclass(frozen=True)
class GroundedProposalTrace:
    """The successful native suffix and terminal message seen by the model."""

    calls: tuple[GroundedToolCall, ...]
    terminal_response: str
    terminal_visible_receipt_refs: tuple[str, ...]
    failed_proposal_call_count: int = 0

    def __post_init__(self) -> None:
        terminal = str(self.terminal_response).strip()
        if not terminal:
            raise ValueError("grounded proposal requires a terminal response")
        if int(self.failed_proposal_call_count) < 0:
            raise ValueError("failed proposal call count must be non-negative")
        node_ids = [call.node_id for call in self.calls]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("grounded proposal node ids must be unique")
        receipt_refs = [call.receipt_ref for call in self.calls]
        if len(receipt_refs) != len(set(receipt_refs)):
            raise ValueError("grounded proposal receipt refs must be unique")

        available: set[str] = set()
        for call in self.calls:
            visible = set(call.visible_receipt_refs)
            if not visible <= available:
                raise ValueError(
                    "grounded call cites a receipt not visible before the call"
                )
            available.add(call.receipt_ref)
        terminal_visible = set(self.terminal_visible_receipt_refs)
        if terminal_visible != available:
            raise ValueError(
                "grounded terminal must be bound to every successful proposal receipt"
            )

    def public_summary(self) -> dict[str, Any]:
        dependency_edge_count = sum(
            len(call.visible_receipt_refs) for call in self.calls
        )
        return {
            "schema": SCHEMA,
            "proposal_mode": "native_tool_receipt_grounded",
            "native_call_node_count": len(self.calls),
            "successful_call_node_count": sum(
                int(call.succeeded) for call in self.calls
            ),
            "failed_call_evidence_node_count": sum(
                int(not call.succeeded) for call in self.calls
            ),
            "receipt_dependency_edge_count": dependency_edge_count,
            "terminal_dependency_count": len(self.terminal_visible_receipt_refs),
            "terminal_bound": True,
            "failed_proposal_call_count": int(self.failed_proposal_call_count),
            "model_observed_receipts": True,
            "blind_typed_plan_model_facing": False,
            "model_authored_relations": False,
            "concrete_arguments_public": False,
            "native_results_public": False,
        }


@dataclass(frozen=True)
class GroundedNodeProjection:
    """Surface-certified public annotations for one recorded native call."""

    authority_label: str
    semantic_effects: tuple[str, ...] = ()
    effect_families: tuple[str, ...] = ()
    covered_obligation_ids: tuple[str, ...] = ()
    evidence_fact_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class GroundedProposalBundle:
    """Shared IR plus private executable bindings for Checked Execution."""

    proposal_graph: ProposalGraph
    bindings: Mapping[str, CFRCNodeBinding] = field(repr=False)
    public_summary: Mapping[str, Any] = field(default_factory=dict)


GroundedCallProjector = Callable[[GroundedToolCall], GroundedNodeProjection]


def bridge_grounded_proposal(
    *,
    trace: GroundedProposalTrace,
    graph_id: str,
    residual_obligation_ids: Sequence[str],
    evidence_fact_ids: Sequence[str],
    terminal_authority_label: str,
    project_call: GroundedCallProjector,
    bind_value_provenance: bool = False,
    frontier_values: frozenset[str] = frozenset(),
    publish_effect_receipts: bool = False,
    include_numeric_value_provenance: bool = False,
    reject_unbound_value_provenance: bool = False,
    declarations: Sequence[GroundedDischargeDeclaration] = (),
    declaration_authority_label: str = "",
    coverage_unit_tokens_by_obligation: Mapping[str, Sequence[str]] | None = None,
) -> GroundedProposalBundle:
    """Mechanically lift a realized native suffix into the shared CFRC IR.

    With ``bind_value_provenance`` the bridge also traces each argument value
    back to the receipt that produced it and records the result as an argument
    carrier, so the compiler enforces that a value created during completion is
    consumed only through a receipt from a selected predecessor.  Only values
    produced by a node with a receipt-supported evidence output become carriers.
    Effect-free calls publish those outputs by default.  Surfaces whose native
    successful writes return values may opt in with ``publish_effect_receipts``;
    the default is false so existing graph and compiler decisions do not move.
    """

    if publish_effect_receipts and not bind_value_provenance:
        raise ValueError(
            "effect receipt publication requires value provenance binding"
        )
    if include_numeric_value_provenance and not bind_value_provenance:
        raise ValueError(
            "numeric value provenance requires value provenance binding"
        )
    if reject_unbound_value_provenance and not bind_value_provenance:
        raise ValueError(
            "rejecting unbound value provenance requires value provenance binding"
        )
    if publish_effect_receipts and any(not call.succeeded for call in trace.calls):
        raise ValueError(
            "failed proposal calls cannot publish receipt-supported outputs"
        )
    obligation_ids = tuple(dict.fromkeys(str(item) for item in residual_obligation_ids))
    if not obligation_ids:
        raise ValueError("grounded proposal requires residual obligations")
    action_receipt_coverage_enabled = coverage_unit_tokens_by_obligation is not None
    coverage_units: dict[str, tuple[str, ...]] = {}
    if coverage_unit_tokens_by_obligation is not None:
        normalized_coverage_units = {
            str(obligation_id).strip(): raw_units
            for obligation_id, raw_units in coverage_unit_tokens_by_obligation.items()
        }
        if len(normalized_coverage_units) != len(coverage_unit_tokens_by_obligation):
            raise ValueError("coverage unit mapping obligation ids must be unique")
        if set(normalized_coverage_units) != set(obligation_ids):
            raise ValueError(
                "coverage unit mapping must match residual obligation ids"
            )
        for obligation_id in obligation_ids:
            units = tuple(
                str(unit).strip()
                for unit in normalized_coverage_units[obligation_id]
            )
            if any(not unit for unit in units):
                raise ValueError("coverage unit tokens must be non-empty")
            if len(units) != len(set(units)):
                raise ValueError("coverage unit tokens must be unique")
            coverage_units[obligation_id] = units
    declared_discharges = _validated_declarations(
        declarations, obligation_ids=obligation_ids
    )
    if not trace.calls and not declared_discharges:
        raise ValueError(
            "grounded proposal requires a native call or a discharge declaration"
        )
    # A declaration exercises no authority, so it must never be what raises a
    # candidate's authority rank. The terminal is present in every candidate the
    # selector can complete, so pinning the declaration to the terminal's label
    # makes that structural rather than a convention a surface has to honor.
    if declaration_authority_label not in ("", terminal_authority_label):
        raise ValueError(
            "discharge declarations must carry the terminal authority label"
        )
    evidence_ids = tuple(dict.fromkeys(str(item) for item in evidence_fact_ids))
    receipt_to_node = {call.receipt_ref: call.node_id for call in trace.calls}
    nodes: list[ProposalNode] = []
    bindings: dict[str, CFRCNodeBinding] = {}
    covered: set[str] = set()
    evidence_producer_ids: set[str] = set()
    prior_rows: list[dict[str, Any]] = []
    bound_provenance: list[ValueProvenance] = []
    unbound_write_source_count = 0
    effect_receipt_output_count = 0

    for call in trace.calls:
        projection = project_call(call)
        call_covered = set(projection.covered_obligation_ids)
        unknown_coverage = call_covered - set(obligation_ids)
        if unknown_coverage:
            raise ValueError(
                "grounded call covers unknown obligations: "
                + ",".join(sorted(unknown_coverage))
            )
        covered.update(call_covered)
        coverage_claims = (
            tuple(
                ProposalCoverageClaim(
                    obligation_id=obligation_id,
                    unit_tokens=coverage_units[obligation_id],
                    replica_receipt_ref=call.receipt_ref,
                )
                for obligation_id in obligation_ids
                if obligation_id in call_covered
            )
            if action_receipt_coverage_enabled
            else ()
        )
        dependency_ids = tuple(
            receipt_to_node[receipt] for receipt in call.visible_receipt_refs
        )
        effects = tuple(dict.fromkeys(projection.semantic_effects))
        carriers: frozenset[str] = frozenset()
        carrier_tags: Mapping[str, str] = {}
        if bind_value_provenance:
            derived = derive_value_provenance(
                arguments=call.arguments,
                prior_calls=prior_rows,
                frontier_values=frontier_values,
                include_numbers=include_numeric_value_provenance,
            )
            # A carrier may only name a node that already publishes a receipt
            # evidence output and that this call already declares as a
            # dependency. Effect-free predecessors publish by default; an
            # effect predecessor publishes only under the explicit opt-in.
            # In either case the compiler sees and checks the same carrier.
            declared = set(dependency_ids)
            usable = tuple(
                record
                for record in derived
                if record.source_node_id in evidence_producer_ids
                and record.source_node_id in declared
            )
            unbound_write_source_count += len(derived) - len(usable)
            if usable:
                bound_provenance.extend(usable)
                candidate = provenance_carrier_units(usable)
                if candidate.isdisjoint(effects):
                    carriers = candidate
                    carrier_tags = provenance_carrier_tags(usable)
        closure = build_executable_closure(
            semantic_effects=frozenset(effects),
            argument_carriers=carriers,
            carrier_provenance=carrier_tags,
        )
        prior_rows.append(
            {
                "node_id": call.node_id,
                "arguments": dict(call.arguments),
                "result": call.result,
            }
        )
        payload = dict(call.replay_payload)
        if not payload:
            payload = {
                "kind": "tool_call",
                "tool_calls": [
                    {
                        "tool_name": call.tool_name,
                        "arguments": dict(call.arguments),
                    }
                ],
            }
        payload["expected_receipt_ref"] = call.receipt_ref
        payload["receipt_instance_id"] = call.node_id
        if effects:
            evidence_output_refs = (
                (carrier_unit_for(call.node_id),)
                if publish_effect_receipts
                else ()
            )
            node = ProposalNode(
                node_id=call.node_id,
                kind="tool_call",
                authority_label=projection.authority_label,
                semantic_effects=effects,
                effect_families=tuple(dict.fromkeys(projection.effect_families)),
                evidence_output_refs=evidence_output_refs,
                dependency_ids=dependency_ids,
                provenance_refs=(call.receipt_ref,),
                evidence_fact_ids=tuple(dict.fromkeys(projection.evidence_fact_ids)),
                coverage_claims=coverage_claims,
                executable_closure=closure,
                tool_call_count=1,
            )
            binding = CFRCNodeBinding(
                node_id=call.node_id,
                executable_closure=closure,
                suffix_payload=payload,
                binding_kind="tool_call",
                evidence_output_refs=evidence_output_refs,
                evidence_receipt_refs=(
                    (call.receipt_ref,) if evidence_output_refs else ()
                ),
                replica_receipt_ref=call.receipt_ref,
                replica_execution_succeeded=call.succeeded,
            )
            if evidence_output_refs:
                evidence_producer_ids.add(call.node_id)
                effect_receipt_output_count += 1
        else:
            evidence_output_ref = carrier_unit_for(call.node_id)
            evidence_producer_ids.add(call.node_id)
            node = ProposalNode(
                node_id=call.node_id,
                kind="tool_call",
                authority_label=projection.authority_label,
                node_role="evidence_acquisition",
                evidence_output_refs=(evidence_output_ref,),
                dependency_ids=dependency_ids,
                provenance_refs=(call.receipt_ref,),
                evidence_fact_ids=tuple(dict.fromkeys(projection.evidence_fact_ids)),
                coverage_claims=coverage_claims,
                executable_closure=closure,
                tool_call_count=1,
            )
            binding = CFRCNodeBinding(
                node_id=call.node_id,
                executable_closure=closure,
                suffix_payload=payload,
                binding_kind="tool_call",
                evidence_output_refs=(evidence_output_ref,),
                evidence_receipt_refs=(call.receipt_ref,),
                replica_receipt_ref=call.receipt_ref,
                replica_execution_succeeded=call.succeeded,
            )
        nodes.append(node)
        bindings[call.node_id] = binding

    if reject_unbound_value_provenance and unbound_write_source_count:
        raise ValueError(
            "grounded proposal contains unbound derived value provenance: "
            f"{unbound_write_source_count}"
        )

    realized_node_ids = tuple(call.node_id for call in trace.calls)
    declared_by_obligation = {
        item.obligation_id: item for item in declared_discharges
    }
    double_covered = sorted(set(declared_by_obligation) & covered)
    if double_covered:
        raise ValueError(
            "obligation is both realized and declared: " + ",".join(double_covered)
        )
    declaration_closure = build_executable_closure(
        semantic_effects=frozenset(),
        argument_carriers=frozenset(),
    )
    declaration_node_ids: list[str] = []
    for index, item in enumerate(declared_discharges, start=1):
        node_id = f"{DECLARATION_NODE_PREFIX}{item.mode}_{index:03d}"
        nodes.append(
            ProposalNode(
                node_id=node_id,
                kind="message",
                node_role="discharge_declaration",
                authority_label=declaration_authority_label
                or terminal_authority_label,
                # The id records which obligation the requested failure return
                # concerns. The selector excludes declaration-role nodes from
                # successful coverage.
                discharged_obligation_ids=(item.obligation_id,),
                # A declaration is only legible against what the successor
                # actually observed, so it sits after every realized receipt.
                dependency_ids=realized_node_ids,
                evidence_fact_ids=evidence_ids,
                executable_closure=declaration_closure,
                tool_call_count=0,
            )
        )
        bindings[node_id] = CFRCNodeBinding(
            node_id=node_id,
            executable_closure=declaration_closure,
            suffix_payload={
                "kind": "declaration",
                "obligation_id": item.obligation_id,
                "mode": item.mode,
                "basis": str(item.basis).strip(),
            },
            binding_kind="message",
        )
        declaration_node_ids.append(node_id)

    missing = set(obligation_ids) - covered
    undeclared_missing = missing - set(declared_by_obligation)
    if undeclared_missing:
        raise ValueError(
            "grounded proposal leaves obligations uncovered: "
            + ",".join(sorted(undeclared_missing))
        )

    terminal_dependencies = tuple(
        receipt_to_node[receipt] for receipt in trace.terminal_visible_receipt_refs
    ) + tuple(declaration_node_ids)
    terminal_closure = build_executable_closure(
        semantic_effects=frozenset(),
        argument_carriers=frozenset(),
    )
    terminal = ProposalNode(
        node_id=TERMINAL_NODE_ID,
        kind="message",
        node_role="terminal_communication",
        authority_label=terminal_authority_label,
        discharged_obligation_ids=(
            ()
            if action_receipt_coverage_enabled
            else tuple(
                obligation_id
                for obligation_id in obligation_ids
                if obligation_id in covered
            )
        ),
        dependency_ids=terminal_dependencies,
        provenance_refs=tuple(trace.terminal_visible_receipt_refs),
        evidence_fact_ids=evidence_ids,
        executable_closure=terminal_closure,
        tool_call_count=0,
    )
    nodes.append(terminal)
    bindings[TERMINAL_NODE_ID] = CFRCNodeBinding(
        node_id=TERMINAL_NODE_ID,
        executable_closure=terminal_closure,
        suffix_payload={
            "kind": "message",
            "assistant_response": str(trace.terminal_response).strip(),
        },
        binding_kind="message",
    )
    graph = ProposalGraph(
        nodes=tuple(nodes),
        graph_id=graph_id,
        residual_obligation_ids=obligation_ids,
        evidence_fact_ids=evidence_ids,
    )
    declaration_modes: dict[str, int] = {}
    for item in declared_discharges:
        declaration_modes[item.mode] = declaration_modes.get(item.mode, 0) + 1
    public_summary = {
        **trace.public_summary(),
        "covered_obligation_count": len(covered),
        "all_obligations_covered": covered == set(obligation_ids),
        "proposal_graph_node_count": len(nodes),
        **_graph_structure_diagnostics(nodes),
        "declaration_node_count": len(declaration_node_ids),
        "declaration_modes": {
            key: declaration_modes[key] for key in sorted(declaration_modes)
        },
        "realized_discharge_obligation_count": len(covered),
        "declared_discharge_obligation_count": len(declared_discharges),
        "action_receipt_coverage_enabled": action_receipt_coverage_enabled,
        "action_receipt_coverage_claim_count": sum(
            len(node.coverage_claims) for node in nodes
        ),
        "value_provenance_binding_enabled": bool(bind_value_provenance),
        **(
            provenance_public_summary(bound_provenance)
            if bind_value_provenance
            else {}
        ),
        # Keep the legacy field name for schema compatibility. Under the
        # opt-in it counts any derived source that still lacks a declared,
        # receipt-supported predecessor output.
        "value_provenance_unbound_source_count": (
            unbound_write_source_count if bind_value_provenance else 0
        ),
        **(
            {"unbound_value_provenance_rejected": True}
            if reject_unbound_value_provenance
            else {}
        ),
        **(
            {
                "effect_receipt_outputs_enabled": True,
                "effect_receipt_output_count": effect_receipt_output_count,
            }
            if publish_effect_receipts
            else {}
        ),
        **(
            {"numeric_value_provenance_enabled": True}
            if include_numeric_value_provenance
            else {}
        ),
    }
    return GroundedProposalBundle(
        proposal_graph=graph,
        bindings=bindings,
        public_summary=public_summary,
    )


def _graph_structure_diagnostics(nodes: Sequence[ProposalNode]) -> dict[str, int]:
    """Report the emitted graph's shape as public-safe integer counts.

    A surface typically makes every receipt seen so far visible to the next call,
    so the declared dependency edges are a transitive closure and overstate the
    real structure: a plain chain of ``n`` nodes declares ``n*(n-1)/2`` edges. The
    counts here describe the same graph after transitive reduction, which is what
    the paper should report.

    ``graph_depth`` counts the nodes on the longest dependency chain.
    ``graph_max_antichain_width`` groups nodes by longest-path level and takes the
    largest level; nodes sharing a level are pairwise incomparable, so this is a
    lower bound on the true maximum antichain width and never overstates how much
    of the residual could run in parallel.

    Diagnostics only: nothing here feeds a coverage, selection, or compile
    decision.
    """

    node_ids = [node.node_id for node in nodes]
    present = set(node_ids)
    predecessors = {
        node.node_id: tuple(
            dict.fromkeys(dep for dep in node.dependency_ids if dep in present)
        )
        for node in nodes
    }
    order = _topological_order(node_ids, predecessors)
    reachable: dict[str, frozenset[str]] = {}
    level: dict[str, int] = {}
    reduced_edge_count = 0
    for node_id in order:
        preds = predecessors[node_id]
        ancestors: set[str] = set()
        for dep in preds:
            ancestors |= reachable.get(dep, frozenset())
        # An edge survives reduction only when the predecessor is not already
        # reachable through another predecessor of the same node.
        reduced_edge_count += sum(1 for dep in preds if dep not in ancestors)
        reachable[node_id] = frozenset(ancestors | set(preds))
        level[node_id] = 1 + max((level.get(dep, 0) for dep in preds), default=0)
    width_by_level: dict[int, int] = {}
    for value in level.values():
        width_by_level[value] = width_by_level.get(value, 0) + 1
    return {
        "graph_transitive_reduction_edge_count": reduced_edge_count,
        "graph_max_antichain_width": max(width_by_level.values(), default=0),
        "graph_depth": max(level.values(), default=0),
    }


def _topological_order(
    node_ids: Sequence[str],
    predecessors: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    """Order nodes so every dependency precedes its dependent.

    The bridge already emits nodes in this order; recomputing it keeps the shape
    counts correct if that emission order ever changes, and leaves any node in a
    cycle out rather than looping.
    """

    remaining = {node_id: set(predecessors[node_id]) for node_id in node_ids}
    ordered: list[str] = []
    settled: set[str] = set()
    progressed = True
    while progressed:
        progressed = False
        for node_id in node_ids:
            if node_id in settled or remaining[node_id] - settled:
                continue
            ordered.append(node_id)
            settled.add(node_id)
            progressed = True
    return tuple(ordered)


def _validated_declarations(
    declarations: Sequence[GroundedDischargeDeclaration],
    *,
    obligation_ids: Sequence[str],
) -> tuple[GroundedDischargeDeclaration, ...]:
    """Keep declarations that name a distinct open obligation of this contract."""

    known = set(obligation_ids)
    seen: set[str] = set()
    items: list[GroundedDischargeDeclaration] = []
    for item in declarations:
        if item.obligation_id not in known:
            raise ValueError(
                "discharge declaration names an unknown obligation: "
                + item.obligation_id
            )
        if item.obligation_id in seen:
            raise ValueError(
                "discharge declaration repeats an obligation: " + item.obligation_id
            )
        seen.add(item.obligation_id)
        items.append(item)
    return tuple(items)


def receipt_ref(
    *,
    tool_name: str,
    arguments: Mapping[str, Any],
    result: Any,
    instance_id: str | None = None,
) -> str:
    """Return an opaque digest for one native call/arguments/result receipt."""

    payload = {
        "tool_name": str(tool_name),
        "arguments": dict(arguments),
        "result": result,
    }
    if instance_id is not None:
        payload["instance_id"] = str(instance_id)
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("grounded proposal values must be JSON serializable") from exc


__all__ = [
    "DECLARATION_NODE_PREFIX",
    "DischargeMode",
    "GroundedDischargeDeclaration",
    "GroundedNodeProjection",
    "GroundedProposalBundle",
    "GroundedProposalTrace",
    "GroundedToolCall",
    "TERMINAL_NODE_ID",
    "bridge_grounded_proposal",
    "receipt_ref",
]

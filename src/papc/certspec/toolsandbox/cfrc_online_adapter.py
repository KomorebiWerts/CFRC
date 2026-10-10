"""ToolSandbox adapter for the shared CFRC online runner.

This module translates public typed-suffix rows into the benchmark-independent
online skeleton seam. It intentionally does not execute ToolSandbox tools or run
the official evaluator; paid online execution must add a runtime adapter around
the same seam instead of creating a separate base129 loop.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from papc.certspec.cfrc_compiler import CFRCFrontier, CFRCNodeBinding
from papc.certspec.cfrc_ir import ProposalGraph, ProposalNode
from papc.certspec.cfrc_online_runner import (
    CFRCOnlineCandidate,
    CFRCOnlineExecution,
    assert_public_summary_safe,
)
from papc.certspec.executable_closure import build_executable_closure
from papc.certspec.state_write_contract import Obligation, StateWriteContract
from papc.certspec.toolsandbox.grounded_residual_authoring import (
    obligation_id_for_unit,
)

TOOLSANDBOX_CFRC_ADAPTER_SCHEMA = "papc.toolsandbox_cfrc_online_adapter.v1"
TOOLSANDBOX_ACTION_CANDIDATE_ID_PREFIX = "toolsandbox_typed_suffix"
MUTATING_TOOL_NAMES = {
    "add_contact",
    "modify_contact",
    "remove_contact",
    "add_reminder",
    "modify_reminder",
    "remove_reminder",
    "send_message_with_phone_number",
    "set_low_battery_mode_status",
    "set_location_service_status",
    "set_cellular_service_status",
    "set_wifi_status",
}


@dataclass
class ToolSandboxTypedSuffixOnlineAdapter:
    """Translate one public typed-suffix row into the shared online skeleton."""

    cheap_messages: Sequence[Mapping[str, Any]]
    typed_row: Mapping[str, Any]
    private_suffix_payload_by_node: Mapping[str, Any] | None = None
    # Whether the residual's user-facing answer arrives as its own terminal node.
    # Receipt-grounded authoring emits one, and the selector must keep it or the
    # answer is never submitted. The single-shot graph protocol instead carries
    # the answer on the acting payload, so it declares no terminal node and this
    # stays off.
    user_facing_message_required: bool = False
    surface: str = "toolsandbox"

    def cheap_conversation(self) -> list[Mapping[str, Any]]:
        return [copy.deepcopy(dict(message)) for message in self.cheap_messages]

    def enumerate_frontier_candidates(
        self, prefix_payload: Mapping[str, Any]
    ) -> Sequence[CFRCOnlineCandidate]:
        candidates: list[CFRCOnlineCandidate] = [
            CFRCOnlineCandidate.preserve(
                candidate_id="preserve",
                frontier_message_index=0,
                preserve_reason="preserve_floor",
            )
        ]
        if not _row_is_actionable(self.typed_row):
            return candidates
        frontier_binding = _frontier_binding(self.typed_row)
        frontier_index = int(frontier_binding["frontier_message_index"])
        conversation = _conversation(prefix_payload)
        if len(conversation) < frontier_index:
            return candidates
        if _prefix_has_mutating_tool(conversation[:frontier_index]):
            return candidates
        candidates.append(
            CFRCOnlineCandidate.action(
                candidate_id=_candidate_id(self.typed_row),
                frontier_kind=str(frontier_binding["frontier_kind"]),
                frontier_message_index=frontier_index,
                frontier_tool_call_index=frontier_binding.get("frontier_tool_call_index"),
                authority_ceiling=str(frontier_binding["authority_ceiling"]),
                effect_units=tuple(str(unit) for unit in frontier_binding["effect_units"]),
            )
        )
        return candidates

    def build_frontier(
        self,
        candidate: CFRCOnlineCandidate,
        prefix_payload: Mapping[str, Any],
    ) -> CFRCFrontier:
        effect_units = tuple(candidate.effect_units)
        if not effect_units:
            raise ValueError("toolsandbox action candidates require effect units")
        obligations = frozenset(
            Obligation(
                obligation_id=obligation_id_for_unit(unit),
                write_footprint=frozenset({unit}),
                postcondition_witness="replica_receipt_for_visible_prefix_residual",
                surface="toolsandbox",
                status="open_bounded",
            )
            for unit in effect_units
        )
        contract = StateWriteContract(
            surface="toolsandbox",
            derivation_timing="prefix",
            residual=obligations,
            legal_actions=frozenset({"bounded_repair", "abstain"}),
            state_change_required=True,
            # The answer the successor owes the user is part of the residual, not
            # a courtesy after it. Without this the selector drops the terminal
            # node from every candidate -- it covers no obligation -- and the
            # final message is never submitted to the surface at all.
            user_facing_message_required=bool(self.user_facing_message_required),
            soundness_assumptions=(
                "toolsandbox_typed_suffix_row_uses_public_schema_shape_only",
                "runtime_selector_does_not_read_scorer_or_gold_state",
            ),
        )
        return CFRCFrontier(
            contract=contract,
            allowed_effect_units=frozenset(effect_units),
            frontier_timing="pre_write",
            # The reached state holds no record field the residual must
            # retransmit, so there is no frontier-supplied carrier. Every carrier
            # this surface uses is produced by a selected suffix node and is
            # verified through that node's receipt instead.
            pending_carrier_refs=(),
        )

    def bind_proposal(
        self,
        candidate: CFRCOnlineCandidate,
        proposal_graph: ProposalGraph,
        prefix_payload: Mapping[str, Any],
    ) -> Mapping[str, CFRCNodeBinding]:
        typed_suffix = _typed_suffix(self.typed_row)
        bindings: dict[str, CFRCNodeBinding] = {}
        private_payloads = self.private_suffix_payload_by_node or {}
        for node in proposal_graph.nodes:
            if node.executable_closure is None:
                continue
            suffix_payload = private_payloads.get(node.node_id)
            if suffix_payload is None:
                suffix_payload = {
                    "template_id": str(typed_suffix["template_id"]),
                    "tool_sequence": list(typed_suffix["suffix_tool_sequence"]),
                    "mutating_tool_name": str(typed_suffix["mutating_tool_name"]),
                    "mutating_arg_key_count": len(
                        typed_suffix.get("mutating_arg_key_types") or []
                    ),
                    "effect_unit_count": len(node.semantic_effects),
                    "fixed_argument_values_used": False,
                }
            bindings[node.node_id] = CFRCNodeBinding(
                node_id=node.node_id,
                executable_closure=node.executable_closure,
                suffix_payload=copy.deepcopy(suffix_payload),
                binding_kind="tool_call",
                carrier_refs=(),
            )
        return bindings

    def proposer_prompt_contract(
        self,
        candidate: CFRCOnlineCandidate,
        frontier: Any,
        prefix_payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        typed_suffix = _typed_suffix(self.typed_row)
        return {
            "typed_suffix": copy.deepcopy(dict(typed_suffix)),
        }

    def execute_suffix(
        self,
        candidate: CFRCOnlineCandidate,
        compile_result: Any,
    ) -> CFRCOnlineExecution:
        payloads = [
            payload for payload in compile_result.suffix_payloads if isinstance(payload, Mapping)
        ]
        tool_call_count = sum(len(payload.get("tool_sequence") or []) for payload in payloads)
        summary = {
            "schema": TOOLSANDBOX_CFRC_ADAPTER_SCHEMA,
            "provider_calls": 0,
            "actual_tool_execution_performed": False,
            "official_evaluator_used_at_runtime": False,
            "suffix_payload_count": len(payloads),
            "suffix_tool_call_count": tool_call_count,
            "mutating_tool_names": sorted(
                {
                    str(payload.get("mutating_tool_name"))
                    for payload in payloads
                    if payload.get("mutating_tool_name")
                }
            ),
            "adapter_mode": "noapi_typed_suffix_smoke",
            "residual_closure_evidence": False,
            "raw_tool_args_present": False,
        }
        assert_public_summary_safe(summary)
        return CFRCOnlineExecution(executed=False, public_summary=summary)


def proposal_graph_from_toolsandbox_typed_row(row: Mapping[str, Any]) -> ProposalGraph:
    """Build a public-safe proposal graph from one typed ToolSandbox row."""

    frontier_binding = _frontier_binding(row)
    typed_suffix = _typed_suffix(row)
    effect_units = tuple(str(unit) for unit in frontier_binding["effect_units"])
    authority = str(frontier_binding["authority_ceiling"])
    node = ProposalNode(
        node_id="toolsandbox_typed_suffix_node",
        kind="tool_call",
        authority_label=authority,
        semantic_effects=effect_units,
        effect_families=_effect_families(
            typed_suffix=typed_suffix,
            effect_units=effect_units,
        ),
        executable_closure=build_executable_closure(
            semantic_effects=frozenset(effect_units),
            argument_carriers=frozenset(),
        ),
        tool_call_count=len(typed_suffix.get("suffix_tool_sequence") or []),
    )
    return ProposalGraph(nodes=(node,), graph_id="toolsandbox_typed_suffix_graph")


def _row_is_actionable(row: Mapping[str, Any]) -> bool:
    return (
        row.get("residual_closable") is True
        and str(row.get("method_introduced_minefield_risk") or "none") == "none"
        and not row.get("typed_suffix_blockers")
        and bool(_frontier_binding(row).get("effect_units"))
    )


def _candidate_id(row: Mapping[str, Any]) -> str:
    return f"{TOOLSANDBOX_ACTION_CANDIDATE_ID_PREFIX}_{int(row['public_ordinal']):03d}"


def _frontier_binding(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("frontier_binding")
    if not isinstance(value, Mapping):
        raise ValueError("typed row missing frontier_binding")
    return value


def _typed_suffix(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("typed_suffix")
    if not isinstance(value, Mapping):
        raise ValueError("typed row missing typed_suffix")
    return value


def _effect_families(
    *,
    typed_suffix: Mapping[str, Any],
    effect_units: Sequence[str],
) -> tuple[str, ...]:
    explicit = tuple(
        str(item)
        for item in typed_suffix.get("effect_families") or ()
        if str(item)
    )
    if explicit:
        return explicit
    inferred = tuple(
        sorted(
            {
                ".".join(unit.split(".")[:2])
                for unit in effect_units
                if unit.startswith("toolsandbox.") and len(unit.split(".")) >= 2
            }
        )
    )
    return inferred or ("toolsandbox.reminder",)


def _conversation(prefix_payload: Mapping[str, Any]) -> list[Any]:
    conversation = prefix_payload.get("conversation")
    return list(conversation) if isinstance(conversation, Sequence) else []


def _prefix_has_mutating_tool(conversation: Sequence[Any]) -> bool:
    for message in conversation:
        if not isinstance(message, Mapping):
            continue
        for tool_call in message.get("tool_calls") or []:
            if not isinstance(tool_call, Mapping):
                continue
            name = _tool_name(tool_call)
            if name in MUTATING_TOOL_NAMES:
                return True
    return False


def _tool_name(tool_call: Mapping[str, Any]) -> str:
    function = tool_call.get("function")
    if isinstance(function, Mapping):
        return str(function.get("name") or "")
    return str(tool_call.get("name") or tool_call.get("function_name") or "")


__all__ = [
    "TOOLSANDBOX_ACTION_CANDIDATE_ID_PREFIX",
    "TOOLSANDBOX_CFRC_ADAPTER_SCHEMA",
    "ToolSandboxTypedSuffixOnlineAdapter",
    "proposal_graph_from_toolsandbox_typed_row",
]

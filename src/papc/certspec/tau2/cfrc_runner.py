"""Clean tau2 CFRC runner primitives.

This module keeps the paper-facing protocol separate from older online
controllers. The intended online shape is:

cheap prefix -> reached state z -> one strong typed proposal graph H(a)
-> deterministic Gamma_z / Pi_K projection -> residual suffix.

The pure functions below are importable without tau2.  The tau2 agent
registration function imports tau2 lazily so unit tests can exercise the method
objects without requiring the benchmark package on ``PYTHONPATH``.
"""

from __future__ import annotations

import copy
import json
import re
import secrets
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from papc.certspec.cfrc_compiler import (
    CFRCCompileConfig,
    CFRCFrontier,
    CFRCNodeBinding,
    compile_cfrc_suffix,
)
from papc.certspec.cfrc_ir import (
    AuthorityLattice,
    Commitment,
    EvidenceFact,
    ProposalCoverageClaim,
    ProposalGraph,
    ProposalNode,
    ResidualObligation,
)
from papc.certspec.contract_projection import compact_tool_schema
from papc.certspec.executable_closure import build_executable_closure
from papc.certspec.grounded_proposal import receipt_ref
from papc.certspec.reversibility import (
    IRREVERSIBLE_DAMAGE_THRESHOLD,
    WriteUnitSignal,
    classify_reversibility,
    looks_like_confirm_lifecycle,
    looks_like_fee_or_penalty,
    looks_like_money_or_points,
    restoration_cost_vector,
)
from papc.certspec.runtime_ledger import RuntimeLedger
from papc.certspec.state_write_contract import (
    Obligation,
    StateWriteContract,
)
from papc.certspec.tau2.request_obligation_contract import (
    CLOSURE_DELTA_SCHEMA,
    ContractCall,
    ContractCoverage,
    RequestObligation,
    RequestScopedContract,
    _assert_argument_values_visible,
    apply_closure_delta,
    canonical_call_arguments,
    closure_delta_output_schema,
    closure_delta_rules,
    collect_visible_values,
    contract_feedback,
    contract_literal_is_carried,
    match_contract_calls,
    parse_request_contract,
    request_contract_prompt,
)
from papc.certspec.tau2.request_obligation_contract import (
    SCHEMA as REQUEST_CONTRACT_SCHEMA,
)
from papc.certspec.tau2.residual_support import (
    json_safe_value,
    usage_with_raw_cache_details,
)

ProposalNodeKind = Literal["tool_call", "read_call", "message", "abstain"]
FrontierTiming = Literal["post_tool_result", "pre_write_candidate"]
PostWriteResidualPolicy = Literal["always", "receipt_error"]
PreWriteResidualPolicy = Literal["always", "first_write_candidate"]
CompilerSetting = Literal["strict_core", "authority_lifted_ell3"]
ContractBuilderVariant = Literal[
    "deterministic_candidate_grounded",
    "model_authored_request_semantics",
]
ContractBuilderExecutor = Literal["successor", "cheap_prefix_model"]
MESSAGE_RESIDUAL_UNIT = "message:user_facing"
READ_EFFECT_PREFIX = "read:"
TOOL_EFFECT_PREFIX = "tool:"
TOOL_CALL_EFFECT_PREFIX = "tool_call:"
CHEAP_FRONTIER_CANDIDATE_EVIDENCE = "cheap_frontier_candidate"
STRICT_COMPILER_SETTING: CompilerSetting = "strict_core"
AUTHORITY_LIFTED_ELL3_COMPILER_SETTING: CompilerSetting = "authority_lifted_ell3"
ELL3_AUTHORITY_LABEL = "ell3_bounded_local_redo"
ELL_UPPER_AUTHORITY_LABEL = "ell_upper_full_agent_or_unclassified"
RETAIL_MINIMAL_INTERVENTION_ADAPTER_POLICY = "retail_minimal_intervention_v3"
RETAIL_MINIMAL_INTERVENTION_CANDIDATE_CARRIER_TOOLS = frozenset(
    {
        "exchange_delivered_order_items",
        "modify_pending_order_items",
        "return_delivered_order_items",
    }
)
REPLICA_PROPOSAL_STATE_ATTRIBUTE = "db"
STRICT_COMPILER_AUTHORITY_LABELS = ("ell0", "ell1", "ell_read", "ell_msg", "ell_write")
_SCHEMA_STRUCTURAL_KEYS = frozenset(
    {
        "additionalproperties",
        "allof",
        "anyof",
        "default",
        "description",
        "enum",
        "items",
        "name",
        "nullable",
        "oneof",
        "parameters",
        "properties",
        "required",
        "title",
        "type",
    }
)
_FORBIDDEN_PUBLIC_SUMMARY_TOKENS: tuple[str, ...] = (
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

CFRC_STRONG_SYSTEM_PROMPT = """\
You are the residual proposer in Trajectory-Constrained Residual Completion.
You do not execute tools or write a final answer. Return JSON only.

Your job:
1. Read the reached-state certificate z and visible evidence.
2. Propose a typed residual graph H(a).
3. Build nodes only from proposal_authoring_contract.typed_action_menu. Do not invent tool names or argument fields. Read support nodes may use support_read_action_menu. A typed message node is allowed only when z explicitly enables message_residual_allowed.
4. Do not repeat committed actions in K(z).
5. Do not invent hidden task ids, gold outcomes, or unobserved arguments.
6. If no menu action can bind every required argument from cited evidence within the fixed suffix budget, abstain with one legal_abstain_conditions entry.

Output schema:
{
  "nodes": [
    {
      "id": "n1",
      "kind": "tool_call",
      "authority_level": "ell_write",
      "tool_name": "tool_name_from_S_res",
      "arguments": {"arg": "value"},
      "claimed_effects": ["effect on the residual obligation"],
      "required_evidence": ["visible_evidence event id"],
      "depends_on": []
    },
    {
      "id": "m1",
      "kind": "message",
      "authority_level": "ell_msg",
      "message": {"content": "only when S_res(z) allows a message residual"},
      "claimed_effects": ["message-only residual obligation discharged"],
      "required_evidence": ["visible_evidence event id"],
      "depends_on": []
    },
    {
      "id": "r1",
      "kind": "read_call",
      "authority_level": "ell_read",
      "tool_name": "visible_read_tool",
      "arguments": {"arg": "value copied from cited evidence"},
      "claimed_effects": ["read dependency support only"],
      "required_evidence": ["visible_evidence event id"],
      "depends_on": []
    }
  ],
  "abstain_reason": ""
}
""".strip()

# A terminal handoff is the cheap model declaring it cannot complete the
# request, which is a commitment frontier the request contract must see.
_HANDOFF_TOOL_NAMES = frozenset({"transfer_to_human_agents"})

CFRC_POST_DECISION_TERMINAL_MESSAGE = (
    "I cannot safely perform additional actions after the projected residual decision."
)
CFRC_EMPTY_MODEL_OUTPUT_TERMINAL_MESSAGE = (
    "I cannot continue because the model returned an empty assistant message."
)


@dataclass(frozen=True)
class CFRCProposalNode:
    """One node in the strong model's typed proposal graph ``H(a)``."""

    node_id: str
    kind: ProposalNodeKind
    tool_name: str = ""
    arguments: Mapping[str, Any] = field(default_factory=dict)
    message: str = ""
    authority_level: str = ""
    depends_on: tuple[str, ...] = ()
    claimed_effects: tuple[str, ...] = ()
    required_evidence: tuple[str, ...] = ()

    @property
    def effect_unit(self) -> str:
        if self.kind == "tool_call":
            return tool_effect_unit(self.tool_name)
        if self.kind == "read_call":
            return read_effect_unit(self.tool_name, self.arguments)
        if self.kind == "message":
            return MESSAGE_RESIDUAL_UNIT
        return f"message:{self.node_id}"

    @property
    def is_read_only(self) -> bool:
        return self.kind == "read_call"

    @property
    def is_write_like(self) -> bool:
        return self.kind in {"tool_call", "message"}

    def public_json(self) -> dict[str, Any]:
        return {
            "id_hash": _stable_public_hash(self.node_id),
            "kind": self.kind,
            "authority_level_hash": (
                _stable_public_hash(self.authority_level)
                if self.authority_level
                else ""
            ),
            "tool_name_hash": (
                _stable_public_hash(self.tool_name)
                if self.kind in {"tool_call", "read_call"} and self.tool_name
                else ""
            ),
            "message_chars": len(self.message) if self.kind == "message" else 0,
            "dependency_count": len(self.depends_on),
            "argument_key_count": len(self.arguments),
            "safe_argument_keys": _public_safe_argument_keys(self.arguments),
            "claimed_effect_count": len(self.claimed_effects),
            "required_evidence_count": len(self.required_evidence),
            "required_evidence_safe_id_count": sum(
                1 for item in self.required_evidence if _is_public_safe_text(item)
            ),
            "read_only_support_node": self.is_read_only,
        }

    def private_json(self) -> dict[str, Any]:
        payload = self.public_json()
        payload["id"] = self.node_id
        payload["authority_level"] = self.authority_level
        payload["tool_name"] = self.tool_name if self.kind in {"tool_call", "read_call"} else ""
        payload["depends_on"] = list(self.depends_on)
        payload["claimed_effects"] = list(self.claimed_effects)
        payload["required_evidence"] = list(self.required_evidence)
        payload["arguments"] = dict(self.arguments)
        if self.kind == "message":
            payload["message"] = self.message
        return payload


@dataclass(frozen=True)
class _ToolCallView:
    id: str = ""
    name: str = ""
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CFRCProposalGraph:
    """Typed proposal graph parsed from the strong proposal."""

    nodes: tuple[CFRCProposalNode, ...] = ()
    abstain_reason: str = ""
    parse_error: str = ""

    def node_by_id(self) -> dict[str, CFRCProposalNode]:
        return {node.node_id: node for node in self.nodes}

    def public_json(self) -> dict[str, Any]:
        return {
            "schema": "papc.tau2.cfrc_proposal_graph.v1",
            "node_count": len(self.nodes),
            "nodes": [node.public_json() for node in self.nodes],
            "has_abstain_reason": bool(self.abstain_reason),
            "parse_error_available": bool(self.parse_error),
            "parse_error_hash": _stable_public_hash(self.parse_error)
            if self.parse_error
            else "",
        }

    def private_json(self) -> dict[str, Any]:
        payload = self.public_json()
        payload["nodes"] = [node.private_json() for node in self.nodes]
        payload["abstain_reason"] = self.abstain_reason
        payload["parse_error"] = self.parse_error
        return payload


@dataclass(frozen=True)
class CFRCResidualCertificate:
    """Public-safe tau2 projection of reached-state certificate ``z``."""

    domain: str
    allowed_residual_tools: tuple[str, ...]
    committed_tools: tuple[str, ...] = ()
    commitments: tuple[Commitment, ...] = ()
    residual_obligations: tuple[ResidualObligation, ...] = ()
    evidence_facts: tuple[EvidenceFact, ...] = ()
    frontier_tool_names: tuple[str, ...] = ()
    evidence_event_count: int = 0
    message_residual_allowed: bool = False
    collision_budget: float = 0.0
    restoration_cost: tuple[tuple[str, float], ...] = ()

    @property
    def residual_units(self) -> frozenset[str]:
        return frozenset(
            unit
            for obligation in self.residual_obligations
            for unit in obligation.required_unit_tokens
        )

    @property
    def explicit_residual_units(self) -> frozenset[str]:
        return frozenset(
            unit
            for obligation in self.residual_obligations
            for unit in obligation.required_unit_tokens
        )

    @property
    def committed_units(self) -> frozenset[str]:
        if self.commitments:
            return frozenset(
                unit
                for commitment in self.commitments
                for unit in (*commitment.unit_tokens, *commitment.effect_families)
            )
        return frozenset(tool_effect_unit(name) for name in self.committed_tools)

    @property
    def committed_specific_units(self) -> frozenset[str]:
        if self.commitments:
            return frozenset(
                unit for commitment in self.commitments for unit in commitment.unit_tokens
            )
        return frozenset(tool_effect_unit(name) for name in self.committed_tools)

    @property
    def explicit_k_r_overlap_units(self) -> frozenset[str]:
        return self.committed_specific_units & self.explicit_residual_units

    @property
    def contract_forbidden_units(self) -> frozenset[str]:
        if self.commitments:
            return frozenset(
                unit for commitment in self.commitments for unit in commitment.unit_tokens
            )
        return frozenset(tool_effect_unit(name) for name in self.committed_tools)

    @property
    def restoration_cost_map(self) -> dict[str, float]:
        return {unit: cost for unit, cost in self.restoration_cost}

    def to_contract(self) -> StateWriteContract:
        residual_rows: list[Obligation] = []
        if self.residual_obligations:
            residual_rows.extend(
                Obligation(
                    obligation_id=obligation.obligation_id,
                    write_footprint=frozenset(obligation.required_unit_tokens),
                    surface=f"tau2:{self.domain}",
                    status="open_bounded",
                )
                for obligation in self.residual_obligations
            )
        residual = frozenset(residual_rows)
        return StateWriteContract(
            surface=f"tau2:{self.domain}",
            derivation_timing="prefix",
            residual=residual,
            forbidden_writes=self.contract_forbidden_units,
            legal_actions=frozenset({"bounded_repair", "abstain"}),
            state_change_required=bool(residual),
            commitment_call_count=len(self.commitments) or len(self.committed_tools),
            commitment_footprint_size=len(self.committed_units),
            soundness_assumptions=(
                "tau2 adapter derives mutating tools from reached visible prefix",
                "hidden task ids and gold outcomes are excluded from z",
            ),
        )

    def public_json(self) -> dict[str, Any]:
        return {
            "schema": "papc.tau2.cfrc_residual_certificate.v1",
            "domain": self.domain,
            "K(z)": sorted(self.committed_units),
            "R(z)": sorted(self.residual_units),
            "S_res(z)": list(self.allowed_residual_tools),
            "explicit_R(z)_count": len(self.explicit_residual_units),
            "explicit_K_R_overlap_count": len(self.explicit_k_r_overlap_units),
            "residual_obligation_ids": [
                obligation.obligation_id for obligation in self.residual_obligations
            ],
            "explicit_residual_obligation_count": len(self.residual_obligations),
            "evidence_fact_ids": [fact.fact_id for fact in self.evidence_facts],
            "message_residual_allowed": self.message_residual_allowed,
            "frontier_tool_names": list(self.frontier_tool_names),
            "evidence_event_count": self.evidence_event_count,
            "collision_budget": self.collision_budget,
            "restoration_cost_present": bool(self.restoration_cost),
            "restoration_cost_unit_count": len(self.restoration_cost),
            "public_safe": True,
        }

    def aggregate_public_json(self) -> dict[str, Any]:
        return {
            "schema": "papc.tau2.cfrc_residual_certificate_summary.v1",
            "domain": self.domain,
            "committed_unit_count": len(self.committed_units),
            "residual_unit_count": len(self.residual_units),
            "residual_tool_count": len(self.allowed_residual_tools),
            "explicit_residual_obligation_count": len(self.residual_obligations),
            "explicit_residual_unit_count": len(self.explicit_residual_units),
            "explicit_K_R_overlap_count": len(self.explicit_k_r_overlap_units),
            "evidence_fact_count": len(self.evidence_facts),
            "message_residual_allowed": self.message_residual_allowed,
            "frontier_tool_count": len(self.frontier_tool_names),
            "evidence_event_count": self.evidence_event_count,
            "collision_budget": self.collision_budget,
            "restoration_cost_present": bool(self.restoration_cost),
            "restoration_cost_unit_count": len(self.restoration_cost),
            "public_safe": True,
        }


@dataclass(frozen=True)
class CFRCProjectionResult:
    """Deterministic ``Gamma_z -> Pi_K`` output for one proposal graph."""

    execute: bool
    selected_nodes: tuple[CFRCProposalNode, ...]
    certificate: CFRCResidualCertificate
    proposal: CFRCProposalGraph
    projection: Mapping[str, Any]
    reason: str

    def suffix_tool_calls(self) -> list[dict[str, Any]]:
        return [
            {
                "name": node.tool_name,
                "arguments": dict(node.arguments),
            }
            for node in self.selected_nodes
            if node.kind == "tool_call"
        ]

    def selected_obligation_ids(self) -> list[str]:
        return [
            str(item)
            for item in _list_value(self.projection.get("selected_obligation_ids"))
            if str(item).strip()
        ]

    def public_json(self) -> dict[str, Any]:
        return {
            "schema": "papc.tau2.cfrc_projection_result.v1",
            "execute": self.execute,
            "reason": self.reason,
            "selected_node_count": len(self.selected_nodes),
            "selected_node_id_hashes": [
                _stable_public_hash(node.node_id) for node in self.selected_nodes
            ],
            "selected_tool_names": _public_selected_tool_names(
                self.selected_nodes,
                allowed_tool_names=set(self.certificate.allowed_residual_tools),
            ),
            "selected_read_tool_name_hashes": [
                _stable_public_hash(node.tool_name)
                for node in self.selected_nodes
                if node.kind == "read_call" and node.tool_name
            ],
            "selected_message_count": len(self.suffix_messages()),
            "selected_write_node_count": sum(
                1 for node in self.selected_nodes if node.kind == "tool_call"
            ),
            "selected_read_node_count": sum(
                1 for node in self.selected_nodes if node.kind == "read_call"
            ),
            "selected_obligation_id_hashes": [
                _stable_public_hash(item) for item in self.selected_obligation_ids()
            ],
            "selected_obligation_count": len(self.selected_obligation_ids()),
            "abstain_reason_available": bool(self.proposal.abstain_reason),
            "private_reached_state_replica": bool(
                self.projection.get("private_reached_state_replica")
            ),
            "incremental_live_receipt_grounding": False,
            "proposal_time_replica_receipt_count": _int_value(
                self.projection.get("proposal_time_replica_receipt_count")
            ),
            "live_replica_receipt_match_required": bool(
                self.projection.get("live_replica_receipt_match_required")
            ),
            "certificate": self.certificate.aggregate_public_json(),
            "proposal": self.proposal.public_json(),
            "projection": _public_projection_json(self.projection),
        }

    def suffix_messages(self) -> list[str]:
        return [
            node.message
            for node in self.selected_nodes
            if node.kind == "message" and node.message.strip()
        ]


def tool_effect_unit(tool_name: str) -> str:
    return f"{TOOL_EFFECT_PREFIX}{str(tool_name).strip()}"


def specific_tool_effect_unit(tool_name: str, arguments: Mapping[str, Any]) -> str:
    signature = _tool_call_signature(tool_name, arguments)
    return f"{TOOL_CALL_EFFECT_PREFIX}{str(tool_name).strip()}:{_stable_public_hash(signature)}"


def _terminal_literal_sentence(literals: Sequence[str]) -> str:
    """Deterministic carrier for contract literals a terminal omitted."""

    return "For your records: " + "; ".join(
        str(item) for item in literals if str(item)
    )


CONTRACT_TRANSCRIPTION_TERMINAL = (
    "The requested updates are done; let me know if there is anything else."
)


def _contract_transcription_proposal_json(
    *,
    contract: RequestScopedContract,
    candidate_calls: Sequence[Mapping[str, Any]],
    include_terminal_message: bool,
) -> str | None:
    """Transcribe an independently grounded contract into the typed graph.

    The withheld candidate carries no write authority.  The frozen contract
    supplies every exact action and dependency that the bridge structures,
    after which the graph still passes through projection, private replica,
    live receipts, the ledger and the checked terminal.  Direct transcription
    is limited to a new scope without a candidate, a handoff frontier, or
    same-tool sibling obligations that contain the exact candidate calls.
    Other candidate/contract mismatches retain the independent proposal step.
    """

    if not contract.obligations:
        return None
    candidate_units = Counter(
        specific_tool_effect_unit(
            str(call.get("name") or ""),
            _mapping_or_none(call.get("arguments")) or {},
        )
        for call in candidate_calls
    )
    obligation_units = Counter(
        specific_tool_effect_unit(item.tool_name, item.arguments)
        for item in contract.obligations
    )
    candidate_tool_names = {
        str(call.get("name") or "") for call in candidate_calls
    }
    obligation_tool_names = {
        item.tool_name for item in contract.obligations
    }
    handoff_frontier = (
        len(candidate_calls) == 1
        and str(candidate_calls[0].get("name") or "")
        == "transfer_to_human_agents"
    )
    same_tool_candidate_subset = bool(candidate_units) and (
        not (candidate_units - obligation_units)
        and obligation_tool_names <= candidate_tool_names
    )
    if (
        candidate_calls
        and not handoff_frontier
        and not same_tool_candidate_subset
    ):
        return None
    node_id_by_obligation = {
        item.obligation_id: f"t{index}"
        for index, item in enumerate(contract.obligations, start=1)
    }
    nodes: list[dict[str, Any]] = []
    for index, item in enumerate(contract.obligations, start=1):
        nodes.append(
            {
                "id": f"t{index}",
                "kind": "tool_call",
                "tool_name": item.tool_name,
                "arguments": _json_copy_mapping(item.arguments),
                "claimed_effects": [
                    f"discharge contract obligation {item.obligation_id}"
                ],
                "required_evidence": ["request_contract"],
                "depends_on": sorted(
                    node_id_by_obligation[dependency]
                    for dependency in item.depends_on
                    if dependency in node_id_by_obligation
                ),
            }
        )
    if include_terminal_message:
        nodes.append(
            {
                "id": "m1",
                "kind": "message",
                "authority_level": "ell_msg",
                "message": {
                    "content": (
                        contract.terminal_response
                        or CONTRACT_TRANSCRIPTION_TERMINAL
                    )
                },
                "claimed_effects": ["confirm the completed contract scope"],
                "required_evidence": ["request_contract"],
                "depends_on": [f"t{len(contract.obligations)}"],
            }
        )
    return json.dumps({"nodes": nodes, "abstain_reason": ""})


def _json_copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), default=str))


def _visible_conversation_text(messages: Sequence[Any]) -> str:
    """User requests and tool receipts: the grounded carrier universe.

    The parser grounds obligation arguments in visible values but not
    required_terminal_literals, so a runner-authored sentence may carry a
    literal only when the literal is already present in the visible
    conversation; anything else keeps the fail-closed return."""

    rows = [
        str(_message_attr(message, "content") or "")
        for message in messages
        if _message_attr_str(message, "role") in {"user", "tool"}
        or _message_attr_str(message, "tool_call_id")
    ]
    return "\n".join(row for row in rows if row)


def _grounded_literal_carrier_text(
    messages: Sequence[Any],
    *,
    derived_evidence: Sequence[Mapping[str, Any]] = (),
    public_policy: str = "",
) -> str:
    """The one carrier universe a terminal literal may be grounded in."""

    arithmetic_receipts = [
        row.get("content")
        for row in derived_evidence
        if str(row.get("kind") or "") == "deterministic_receipt_derivation"
    ]
    return "\n".join(
        row
        for row in (
            _visible_conversation_text(messages),
            str(public_policy or ""),
            json.dumps(
                arithmetic_receipts,
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        if row
    )


def _literals_visibly_grounded(
    literals: Sequence[str],
    messages: Sequence[Any],
    *,
    derived_evidence: Sequence[Mapping[str, Any]] = (),
    public_policy: str = "",
) -> bool:
    visible = _grounded_literal_carrier_text(
        messages,
        derived_evidence=derived_evidence,
        public_policy=public_policy,
    )
    return bool(literals) and all(
        contract_literal_is_carried(str(item), visible) for item in literals
    )


def read_effect_unit(tool_name: str, arguments: Mapping[str, Any]) -> str:
    signature = _tool_call_signature(tool_name, arguments)
    return f"{READ_EFFECT_PREFIX}{str(tool_name).strip()}:{_stable_public_hash(signature)}"


def normalize_tool_names(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({str(value).strip() for value in values if str(value).strip()}))


def _compiler_setting(value: Any) -> CompilerSetting:
    if str(value).strip() == AUTHORITY_LIFTED_ELL3_COMPILER_SETTING:
        return AUTHORITY_LIFTED_ELL3_COMPILER_SETTING
    return STRICT_COMPILER_SETTING


def _normalize_restoration_cost(
    values: Mapping[str, float] | Iterable[tuple[str, float]],
) -> tuple[tuple[str, float], ...]:
    items = values.items() if isinstance(values, Mapping) else values
    normalized: list[tuple[str, float]] = []
    for raw_unit, raw_cost in items:
        unit = str(raw_unit).strip()
        if not unit:
            continue
        normalized.append((unit, float(raw_cost)))
    return tuple(sorted(normalized))


def build_residual_certificate(
    *,
    domain: str,
    allowed_residual_tools: Iterable[str],
    committed_tools: Iterable[str] = (),
    commitments: Iterable[Commitment | Mapping[str, Any]] = (),
    residual_obligations: Iterable[ResidualObligation | Mapping[str, Any]] = (),
    evidence_facts: Iterable[EvidenceFact | Mapping[str, Any]] = (),
    frontier_tool_names: Iterable[str] = (),
    evidence_event_count: int = 0,
    message_residual_allowed: bool = False,
    collision_budget: float = 0.0,
    restoration_cost: Mapping[str, float] | Iterable[tuple[str, float]] = (),
) -> CFRCResidualCertificate:
    return CFRCResidualCertificate(
        domain=str(domain),
        allowed_residual_tools=normalize_tool_names(allowed_residual_tools),
        committed_tools=normalize_tool_names(committed_tools),
        commitments=tuple(_coerce_commitment(item) for item in commitments),
        residual_obligations=tuple(
            _coerce_residual_obligation(item) for item in residual_obligations
        ),
        evidence_facts=tuple(_coerce_evidence_fact(item) for item in evidence_facts),
        frontier_tool_names=normalize_tool_names(frontier_tool_names),
        evidence_event_count=max(0, int(evidence_event_count)),
        message_residual_allowed=bool(message_residual_allowed),
        collision_budget=max(0.0, float(collision_budget)),
        restoration_cost=_normalize_restoration_cost(restoration_cost),
    )


def parse_typed_proposal(raw: str | Mapping[str, Any]) -> CFRCProposalGraph:
    if isinstance(raw, Mapping):
        payload = dict(raw)
    else:
        try:
            payload = _json_object_from_text(raw)
        except ValueError as exc:
            return CFRCProposalGraph(parse_error=str(exc))

    nodes: list[CFRCProposalNode] = []
    raw_nodes = payload.get("nodes")
    if not isinstance(raw_nodes, list):
        return CFRCProposalGraph(parse_error="nodes_missing_or_not_list")

    for index, raw_node in enumerate(raw_nodes):
        if not isinstance(raw_node, Mapping):
            continue
        kind = str(raw_node.get("kind") or "tool_call").strip()
        if kind not in {"tool_call", "read_call", "message", "abstain"}:
            continue
        node_id = str(raw_node.get("id") or f"n{index + 1}").strip()
        if not node_id:
            node_id = f"n{index + 1}"
        depends_on = tuple(
            str(dep).strip()
            for dep in _list_value(raw_node.get("depends_on"))
            if str(dep).strip()
        )
        arguments = raw_node.get("arguments")
        arguments = dict(arguments) if isinstance(arguments, Mapping) else {}
        nodes.append(
            CFRCProposalNode(
                node_id=node_id,
                kind=kind,  # type: ignore[arg-type]
                tool_name=str(raw_node.get("tool_name") or "").strip(),
                arguments=arguments,
                message=_message_node_content(raw_node, arguments),
                authority_level=str(
                    raw_node.get("authority_level")
                    or (
                        "ell_msg"
                        if kind == "message"
                        else ("ell_read" if kind == "read_call" else "ell_write")
                    )
                ).strip(),
                depends_on=depends_on,
                claimed_effects=tuple(
                    str(item).strip()
                    for item in _list_value(raw_node.get("claimed_effects"))
                    if str(item).strip()
                ),
                required_evidence=tuple(
                    str(item).strip()
                    for item in _list_value(raw_node.get("required_evidence"))
                    if str(item).strip()
                ),
            )
        )
    return CFRCProposalGraph(
        nodes=tuple(nodes),
        abstain_reason=str(payload.get("abstain_reason") or "").strip(),
    )


def project_cfrc_proposal(
    *,
    certificate: CFRCResidualCertificate,
    proposal: CFRCProposalGraph,
    available_tool_names: Iterable[str],
    required_carrier_tool_calls: Sequence[Mapping[str, Any]] = (),
    visible_evidence: Sequence[Mapping[str, Any]] = (),
    require_full_discharge: bool = True,
    compiler_suffix_tool_call_budget: int | None = None,
    compiler_setting: CompilerSetting = STRICT_COMPILER_SETTING,
    replica_witnesses: Mapping[str, tuple[str, bool]] | None = None,
    coverage_by_node: Mapping[str, Sequence[str]] | None = None,
    require_action_receipt_witness: bool = False,
    parameter_grounding_texts: Mapping[str, Mapping[str, str]] = MappingProxyType({}),
) -> CFRCProjectionResult:
    """Apply deterministic C_z/A_z/Gamma_z/Pi_K over a parsed proposal graph."""

    if proposal.parse_error:
        return _abstain(
            certificate,
            proposal,
            reason="proposal_parse_error",
            projection={
                "decision": "project_abstain",
                "diagnostics": {
                    "schema": "papc.tau2.cfrc_projection_diagnostics.v1",
                    "parse_error_available": True,
                    "parse_error_hash": _stable_public_hash(proposal.parse_error),
                },
            },
        )
    if not certificate.residual_units:
        return _abstain(
            certificate,
            proposal,
            reason="empty_residual_space",
            projection={
                "decision": "project_abstain",
                "diagnostics": {
                    "schema": "papc.tau2.cfrc_projection_diagnostics.v1",
                    "abstain_bucket": "empty_residual_space",
                },
            },
        )

    available = set(normalize_tool_names(available_tool_names))
    allowed = set(certificate.allowed_residual_tools)
    carrier_counts = _carrier_signature_counts(required_carrier_tool_calls)
    structurally_valid_nodes = []
    for node in proposal.nodes:
        reject_reasons = _typed_node_reject_reasons(
            node,
            available_tool_names=available,
            allowed_tool_names=allowed,
            message_residual_allowed=certificate.message_residual_allowed,
            visible_evidence=visible_evidence,
            carrier_tool_calls=required_carrier_tool_calls,
            parameter_grounding_texts=parameter_grounding_texts,
        )
        if reject_reasons:
            continue
        structurally_valid_nodes.append(node)
    structurally_valid_nodes = tuple(structurally_valid_nodes)
    valid_nodes, frontier_binding = _filter_nodes_by_frontier_binding(
        structurally_valid_nodes,
        carrier_counts=carrier_counts,
        carrier_tool_calls=required_carrier_tool_calls,
        visible_evidence=visible_evidence,
    )
    diagnostics = _projection_diagnostics(
        certificate=certificate,
        proposal=proposal,
        available_tool_names=available,
        allowed_tool_names=allowed,
        frontier_binding=frontier_binding,
        visible_evidence=visible_evidence,
        carrier_tool_calls=required_carrier_tool_calls,
        parameter_grounding_texts=parameter_grounding_texts,
    )
    if not valid_nodes:
        return _abstain(
            certificate,
            proposal,
            reason="no_valid_typed_residual_tool_node",
            projection={
                "decision": "project_abstain",
                "diagnostics": diagnostics,
                "carrier_exact_match_required": bool(carrier_counts),
            },
        )

    node_write_units = {
        node.node_id: _node_write_units(
            node,
            certificate=certificate,
            carrier_tool_calls=required_carrier_tool_calls,
        )
        for node in valid_nodes
    }
    proposal_graph, bindings = _shared_compiler_graph_and_bindings(
        certificate=certificate,
        nodes=valid_nodes,
        node_write_units=node_write_units,
        required_carrier_tool_calls=required_carrier_tool_calls,
        replica_witnesses=replica_witnesses or {},
        coverage_by_node=coverage_by_node or {},
    )
    frontier = _shared_compiler_frontier(
        certificate=certificate,
        required_carrier_tool_calls=required_carrier_tool_calls,
        proposal_graph=proposal_graph,
    )
    compile_config = _shared_compiler_config(
        certificate=certificate,
        nodes=valid_nodes,
        compiler_suffix_tool_call_budget=compiler_suffix_tool_call_budget,
        compiler_setting=compiler_setting,
        require_action_receipt_witness=require_action_receipt_witness,
    )
    compile_result = compile_cfrc_suffix(
        frontier=frontier,
        proposal_graph=proposal_graph,
        bindings=bindings,
        config=compile_config,
    )
    projection = compile_result.projection
    projection_json = (
        projection.to_json()
        if projection is not None
        else {
            "decision": (
                "project_block"
                if compile_result.decision == "block"
                else "project_abstain"
            )
        }
    )
    projection_json["diagnostics"] = diagnostics
    projection_json["carrier_exact_match_required"] = bool(carrier_counts)
    projection_json["compiler"] = compile_result.to_json()
    projection_json["legacy_require_full_discharge_argument_observed"] = bool(
        require_full_discharge
    )
    projection_json[
        "shared_compiler_full_discharge_policy"
    ] = "complete_residual_discharge_enforced_by_compiler"
    projection_json["shared_frontier"] = frontier.to_json()
    projection_json["shared_prompt_contract"] = {
        "compiler_setting": _compiler_setting(compiler_setting),
        "mode": compile_config.mode,
        "authority_lattice_labels": [
            level.label for level in compile_config.authority_lattice.levels
        ],
        "allowed_authorities": sorted(compile_config.allowed_authorities),
        "strict_allowed_authorities": sorted(compile_config.strict_allowed_authorities),
        "rejected_authorities": sorted(compile_config.rejected_authorities),
        "collision_budget": compile_config.collision_budget,
        "suffix_tool_call_budget": compile_config.suffix_tool_call_budget,
        "require_action_receipt_witness": (
            compile_config.require_action_receipt_witness
        ),
        "default_off": _compiler_setting(compiler_setting)
        != AUTHORITY_LIFTED_ELL3_COMPILER_SETTING,
    }
    projection_json["shared_graph"] = {
        "stable_node_ids": list(proposal_graph.stable_node_ids()),
        "node_count": len(proposal_graph.nodes),
        "binding_count": len(bindings),
    }
    selected_units = set(projection.projected_footprint) if projection is not None else set()
    projection_json["selected_obligation_ids"] = sorted(
        _selected_obligation_ids(
            valid_nodes,
            node_write_units=node_write_units,
            selected_units=selected_units,
            certificate=certificate,
        )
    )
    adapter_guard_errors = _adapter_ell3_execution_guard_errors(
        certificate=certificate,
        compile_result=compile_result,
    )
    if adapter_guard_errors:
        projection_json["adapter_execution_guard"] = {
            "decision": "block",
            "compile_errors": list(adapter_guard_errors),
            "reason": "authority_lifted_ell3_requires_verified_bounded_local_redo_metadata",
        }
        return CFRCProjectionResult(
            execute=False,
            selected_nodes=(),
            certificate=certificate,
            proposal=proposal,
            projection=projection_json,
            reason="projection_blocked",
        )
    if compile_result.decision != "execute" or projection is None:
        return CFRCProjectionResult(
            execute=False,
            selected_nodes=(),
            certificate=certificate,
            proposal=proposal,
            projection=projection_json,
            reason=(
                "projection_blocked"
                if compile_result.decision == "block"
                else "projection_abstain"
            ),
        )
    if not projection.makes_progress or not selected_units:
        return CFRCProjectionResult(
            execute=False,
            selected_nodes=(),
            certificate=certificate,
            proposal=proposal,
            projection=projection_json,
            reason="projection_abstain",
        )
    selected_ids = _selected_node_ids_with_terminal_completion(
        valid_nodes,
        compiler_selected_ids=set(compile_result.selected_node_ids),
        include_message_dependents=certificate.message_residual_allowed,
    )
    selected_message_ids = {
        node.node_id for node in valid_nodes if node.kind == "message"
    } & selected_ids
    projection_json["terminal_completion_message_policy"] = {
        "enabled": bool(certificate.message_residual_allowed),
        "selected_dependent_message_count": len(selected_message_ids),
        "message_nodes_do_not_increase_tool_suffix_budget": True,
    }
    selected_list: list[CFRCProposalNode] = []
    seen_selected_message = False
    for node in _toposort_nodes(valid_nodes):
        if node.node_id not in selected_ids:
            continue
        if node.kind == "message":
            if seen_selected_message:
                continue
            seen_selected_message = True
        selected_list.append(node)
    selected = tuple(selected_list)
    return CFRCProjectionResult(
        execute=True,
        selected_nodes=selected,
        certificate=certificate,
        proposal=proposal,
        projection=projection_json,
        reason="projected_residual_suffix",
    )


def strong_call_required(certificate: CFRCResidualCertificate) -> bool:
    return bool(certificate.residual_units)


def post_cfrc_cheap_control_blocked(
    *,
    residual_phase_closed: bool,
    pending_suffix_ack: str,
) -> bool:
    """Return whether CFRC must stop instead of re-entering cheap control.

    The online method is a bounded strong residual inserted into the cheap
    trajectory, not a terminal replacement for the rest of the conversation.
    Once the one allowed strong residual attempt is spent, control returns to
    the cheap agent from the current reached state.
    """

    del residual_phase_closed, pending_suffix_ack
    return False


def _tool_schema_name(schema: Mapping[str, Any]) -> str:
    name = schema.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    function = schema.get("function")
    if isinstance(function, Mapping):
        name = function.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return ""


def _tool_schema_parameters(schema: Mapping[str, Any]) -> Mapping[str, Any]:
    function = schema.get("function")
    if isinstance(function, Mapping):
        parameters = function.get("parameters")
        if isinstance(parameters, Mapping):
            return parameters
    parameters = schema.get("parameters")
    if isinstance(parameters, Mapping):
        return parameters
    input_schema = schema.get("input_schema")
    if isinstance(input_schema, Mapping):
        return input_schema
    return {}


def _schema_type_summary(value: Any) -> str | list[str] | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, str):
        names = [str(item).strip() for item in value if str(item).strip()]
        if names:
            return names
    return None


def _public_enum_summary(value: Any) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    rows: list[Any] = []
    for item in value:
        if not isinstance(item, (str, int, float, bool)) and item is not None:
            continue
        if isinstance(item, str) and len(item) > 80:
            continue
        rows.append(item)
    return rows


def _public_schema_property_summary(property_schema: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    schema_type = _schema_type_summary(property_schema.get("type"))
    if schema_type is not None:
        summary["type"] = schema_type
    enum = _public_enum_summary(property_schema.get("enum"))
    if enum:
        summary["enum"] = enum
    items = property_schema.get("items")
    if isinstance(items, Mapping):
        item_type = _schema_type_summary(items.get("type"))
        if item_type is not None:
            summary["items_type"] = item_type
        item_properties = items.get("properties")
        if isinstance(item_properties, Mapping):
            summary["item_property_keys"] = sorted(str(key) for key in item_properties)
    properties = property_schema.get("properties")
    if isinstance(properties, Mapping):
        summary["property_keys"] = sorted(str(key) for key in properties)
    return summary


def _json_schema_argument_contract(schema: Mapping[str, Any]) -> dict[str, Any]:
    parameters = _tool_schema_parameters(schema)
    raw_properties = parameters.get("properties")
    properties = raw_properties if isinstance(raw_properties, Mapping) else {}
    required = [
        str(item)
        for item in parameters.get("required", [])
        if isinstance(item, str) and item in properties
    ]
    optional = sorted(str(key) for key in properties if str(key) not in set(required))
    additional_properties = parameters.get("additionalProperties", True)
    return {
        "required_argument_keys": required,
        "optional_argument_keys": optional,
        "properties": {
            str(key): _public_schema_property_summary(value)
            for key, value in properties.items()
            if isinstance(value, Mapping)
        },
        "additional_properties_allowed": (
            additional_properties if isinstance(additional_properties, bool) else True
        ),
        "field_rule": (
            "Use only schema-declared argument keys unless additionalProperties is true."
        ),
    }


def _typed_action_menu(
    tool_schemas: Sequence[Mapping[str, Any]],
    *,
    kind: ProposalNodeKind,
    authority_level: str,
    frontier_candidate_is_verified_carrier: bool,
) -> list[dict[str, Any]]:
    menu: list[dict[str, Any]] = []
    evidence_sources = ["cited_visible_evidence"]
    if frontier_candidate_is_verified_carrier:
        evidence_sources.append("verified_frontier_candidate_carrier")
    template_evidence = (
        [CHEAP_FRONTIER_CANDIDATE_EVIDENCE]
        if frontier_candidate_is_verified_carrier and kind == "tool_call"
        else ["visible_evidence:<event_id>"]
    )
    for schema in tool_schemas:
        name = _tool_schema_name(schema)
        if not name:
            continue
        argument_contract = _json_schema_argument_contract(schema)
        required = argument_contract["required_argument_keys"]
        menu.append(
            {
                "kind": kind,
                "authority_level": authority_level,
                "tool_name": name,
                "argument_contract": argument_contract,
                "evidence_binding_rule": {
                    "required_arguments_must_be_copied_from": list(evidence_sources),
                    "each_argument_value_needs_cited_evidence": True,
                    "invented_values_forbidden": True,
                },
                "node_template": {
                    "id": "n1" if kind == "tool_call" else "r1",
                    "kind": kind,
                    "authority_level": authority_level,
                    "tool_name": name,
                    "arguments": {
                        key: f"<{key}_copied_from_cited_evidence>"
                        for key in required
                    },
                    "claimed_effects": [
                        (
                            f"discharge residual via {name}"
                            if kind == "tool_call"
                            else f"support residual binding via {name}"
                        )
                    ],
                    "required_evidence": list(template_evidence),
                    "depends_on": [],
                },
            }
        )
    return menu


def _proposal_authoring_contract(
    *,
    residual_tools: Sequence[Mapping[str, Any]],
    support_read_tools: Sequence[Mapping[str, Any]],
    frontier_candidate_is_verified_carrier: bool,
    message_residual_allowed: bool,
    suffix_tool_call_budget: int,
) -> dict[str, Any]:
    message_menu = (
        [
            {
                "kind": "message",
                "authority_level": "ell_msg",
                "depends_on_tool_suffix_required": True,
                "message_must_be_supported_by_visible_evidence": True,
                "does_not_increase_tool_suffix_budget": True,
                "node_template": {
                    "id": "m1",
                    "kind": "message",
                    "authority_level": "ell_msg",
                    "message": {
                        "content": (
                            "<concise confirmation copied from selected suffix and "
                            "visible evidence>"
                        )
                    },
                    "claimed_effects": ["confirm residual completion to the user"],
                    "required_evidence": ["visible_evidence:<event_id>"],
                    "depends_on": ["n1"],
                },
            }
        ]
        if message_residual_allowed
        else []
    )
    return {
        "schema": "papc.tau2.proposal_authoring_contract.v1",
        "typed_action_menu": _typed_action_menu(
            residual_tools,
            kind="tool_call",
            authority_level="ell_write",
            frontier_candidate_is_verified_carrier=frontier_candidate_is_verified_carrier,
        ),
        "support_read_action_menu": _typed_action_menu(
            support_read_tools,
            kind="read_call",
            authority_level="ell_read",
            frontier_candidate_is_verified_carrier=False,
        ),
        "typed_message_action_menu": message_menu,
        "suffix_budget": {
            "write_tool_call_budget": max(0, int(suffix_tool_call_budget)),
            "fixed_by_runner": True,
            "strong_may_not_increase_budget": True,
        },
        "output_construction_rules": [
            "Choose tool_call nodes only from typed_action_menu.",
            "Choose read_call nodes only from support_read_action_menu.",
            "Do not invent tool names, argument keys, or argument values.",
            "Every required argument must be copied from cited visible evidence or an enabled verified frontier carrier.",
            "Each node must cite required_evidence rows that bind its arguments.",
            "Emit the smallest dependency-closed graph within the fixed suffix budget.",
            (
                "When message residual is enabled and a tool_call suffix is proposed, "
                "include one dependent message node from typed_message_action_menu "
                "to close the user-facing terminal completion."
            ),
            "If no legal node can be bound, return no nodes and set abstain_reason to one legal_abstain_conditions entry.",
        ],
        "legal_abstain_conditions": [
            "missing_required_argument_binding",
            "no_visible_evidence_for_required_entity",
            "tool_or_field_not_in_action_menu",
            "extra_read_required_without_support_read_node",
            "suffix_budget_exceeded",
            "would_touch_committed_or_forbidden_effect",
            "verified_carrier_exact_match_unavailable",
            "requires_hidden_benchmark_or_reference_data",
        ],
    }


def build_strong_prompt_payload(
    *,
    certificate: CFRCResidualCertificate,
    visible_evidence: Sequence[Mapping[str, Any]],
    residual_tools: Sequence[Mapping[str, Any]],
    support_read_tools: Sequence[Mapping[str, Any]] = (),
    cheap_frontier_tool_calls: Sequence[Mapping[str, Any]] = (),
    frontier_candidate_is_verified_carrier: bool = False,
    suffix_tool_call_budget: int = 1,
) -> dict[str, Any]:
    """Build the private strong-proposer input payload.

    The caller may persist only :meth:`CFRCResidualCertificate.public_json`; this
    payload may contain visible argument values needed for online execution.
    """

    instruction = (
        "Return exactly one JSON object matching the system schema. "
        "Tool arguments must be copied only from compact visible_evidence rows."
    )
    if cheap_frontier_tool_calls and frontier_candidate_is_verified_carrier:
        instruction = (
            "Return exactly one JSON object matching the system schema. "
            "Tool arguments must be copied only from compact visible_evidence rows "
            "or from verifier_accepted_frontier_candidate carriers. The frontier "
            "candidate is an unexecuted action carrier, not a final answer and not "
            "an escalation decision; cite cheap_frontier_candidate in required_evidence "
            "when you reuse its carrier arguments. Additional sibling tool-call nodes "
            "are allowed only when they use the same tool and argument schema as the "
            "verified carrier and every argument value is copied from cited visible "
            "evidence or the verified carrier."
        )
    authoring_contract = _proposal_authoring_contract(
        residual_tools=residual_tools,
        support_read_tools=support_read_tools,
        frontier_candidate_is_verified_carrier=bool(
            cheap_frontier_tool_calls and frontier_candidate_is_verified_carrier
        ),
        message_residual_allowed=certificate.message_residual_allowed,
        suffix_tool_call_budget=suffix_tool_call_budget,
    )

    return {
        "schema": "papc.tau2.cfrc_strong_prompt_payload.v1",
        "method_chain": "tau_c,s_c -> z -> H(a) -> Gamma_z -> Pi_K -> suffix",
        "z": certificate.public_json(),
        "visible_evidence": list(visible_evidence),
        "residual_tools": list(residual_tools),
        "support_read_tools": list(support_read_tools),
        "proposal_authoring_contract": authoring_contract,
        "message_residual": {
            "enabled": certificate.message_residual_allowed,
            "authority_level": "ell_msg",
            "must_be_typed_graph_node": True,
            "direct_final_answer_forbidden": True,
            "tool_suffix_completion_message_supported": bool(
                certificate.message_residual_allowed
            ),
        },
        "cheap_frontier_candidate": list(cheap_frontier_tool_calls),
        "frontier_candidate_contract": {
            "schema": "papc.tau2.frontier_candidate_contract.v1",
            "is_verifier_accepted_action_carrier": bool(
                cheap_frontier_tool_calls and frontier_candidate_is_verified_carrier
            ),
            "cheap_can_decide_escalation": False,
            "candidate_is_not_final_answer": True,
            "evidence_bound_sibling_rule": {
                "enabled": bool(
                    cheap_frontier_tool_calls and frontier_candidate_is_verified_carrier
                ),
                "same_tool_as_verified_carrier": True,
                "same_argument_schema_as_verified_carrier": True,
                "argument_values_must_come_from": [
                    "cited_visible_evidence",
                    "verified_frontier_candidate_carrier",
                ],
                "if_multiple_siblings_are_supported_return_separate_nodes": True,
            },
        },
        "instruction": instruction,
    }


def mutating_tool_names_by_heuristic(tool_names: Iterable[str]) -> tuple[str, ...]:
    """Conservative fallback when the tau2 toolkit metadata is unavailable."""

    markers = (
        "apply",
        "book",
        "cancel",
        "change",
        "create",
        "exchange",
        "grant",
        "log",
        "modify",
        "process",
        "remove",
        "return",
        "submit",
        "transfer",
        "update",
    )
    return normalize_tool_names(
        name for name in tool_names if str(name).startswith(markers)
    )


def write_jsonl_append(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def cfrc_runtime_ledger_from_results(
    *,
    results_payload: Mapping[str, Any],
    cheap_model: str,
    strong_model: str,
    default_assistant_role: Literal["cheap", "strong"] | None = None,
    sidecar_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    ledger = RuntimeLedger()
    cheap_marker = str(cheap_model or "").lower()
    strong_marker = str(strong_model or "").lower()
    for simulation in _list_value(results_payload.get("simulations")):
        if not isinstance(simulation, Mapping):
            continue
        for message in _list_value(simulation.get("messages")):
            if not isinstance(message, Mapping):
                continue
            raw_data = _mapping_or_none(message.get("raw_data")) or {}
            has_cfrc_control_metadata = (
                _mapping_or_none(raw_data.get("papc_cfrc")) is not None
            )
            role = _runtime_role_for_tau_message(
                message,
                cheap_model=cheap_marker,
                strong_model=strong_marker,
            )
            if role is None and default_assistant_role is not None:
                if (
                    str(message.get("role") or "").lower() == "assistant"
                    and not has_cfrc_control_metadata
                    and _mapping_or_none(message.get("usage")) is not None
                ):
                    role = default_assistant_role
            if role is None:
                continue
            ledger.add_usage(
                role,
                usage_with_raw_cache_details(message),
                latency_seconds=_optional_float(message.get("generation_time_seconds")),
                cost_usd=_optional_float(message.get("cost")),
            )
        evaluation_time = _first_float(
            simulation.get("benchmark_evaluator_time_seconds"),
            simulation.get("evaluation_time_seconds"),
            simulation.get("evaluation_duration_seconds"),
            simulation.get("evaluator_duration_seconds"),
        )
        evaluation_cost = _optional_float(simulation.get("evaluation_cost"))
        reward_info = _mapping_or_none(simulation.get("reward_info"))
        if reward_info is not None or evaluation_time is not None or evaluation_cost is not None:
            ledger.add_usage(
                "evaluator",
                None,
                latency_seconds=evaluation_time,
                cost_usd=evaluation_cost,
            )
    for row in sidecar_rows:
        ledger.merge_public_summary(_mapping_or_none(row.get("runtime_ledger")))
    return ledger.public_summary()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def register_tau2_cfrc_agent(
    *,
    agent_name: str,
    domain: str,
    strong_model: str,
    strong_llm_args: Mapping[str, Any],
    decision_log_path: Path,
    mutating_tool_names: Iterable[str] | None = None,
    include_frontier_candidate_hint: bool = True,
    frontier_timing: FrontierTiming = "post_tool_result",
    post_write_residual_policy: PostWriteResidualPolicy = "always",
    pre_write_residual_policy: PreWriteResidualPolicy = "always",
    strong_call_cap: int = 1,
    suffix_tool_call_budget: int = 1,
    allow_message_residual: bool = False,
    compiler_setting: CompilerSetting = STRICT_COMPILER_SETTING,
    request_contract_verifier: bool = False,
    request_contract_private_replica: bool = False,
    max_request_contract_calls: int = 1,
    request_contract_suffix_cap: int = 6,
    contract_builder_variant: ContractBuilderVariant = (
        "deterministic_candidate_grounded"
    ),
    contract_builder_attempt_cap: int = 2,
    contract_builder_executor: ContractBuilderExecutor = "successor",
    contract_builder_model: str | None = None,
    contract_builder_llm_args: Mapping[str, Any] | None = None,
    deterministic_contract_closure: bool = False,
    zero_obligation_request_fallback: bool = False,
    grounded_terminal_literals: bool = False,
    transcription_is_fallback: bool = False,
) -> str:
    """Register a clean CFRC half-duplex tau2 agent and return its name."""

    if int(max_request_contract_calls) < 1:
        raise ValueError("max_request_contract_calls must be positive")
    if int(request_contract_suffix_cap) < 1:
        raise ValueError("request_contract_suffix_cap must be positive")
    if contract_builder_variant not in {
        "deterministic_candidate_grounded",
        "model_authored_request_semantics",
    }:
        raise ValueError("contract_builder_variant_invalid")
    if int(contract_builder_attempt_cap) not in {1, 2}:
        raise ValueError("contract_builder_attempt_cap_invalid")
    if contract_builder_executor not in {"successor", "cheap_prefix_model"}:
        raise ValueError("contract_builder_executor_invalid")
    if zero_obligation_request_fallback and (
        contract_builder_variant != "deterministic_candidate_grounded"
    ):
        # The fallback is the empty-grounding branch of the candidate-grounded
        # builder. Under the model-authored builder there is no candidate to
        # ground against, so the branch has no meaning.
        raise ValueError("zero_obligation_request_fallback_requires_grounded")

    # Reached-State Contract Construction describes the accepted prefix; it is
    # a distinct component from Executable Residual Completion and does not
    # have to be discharged by the successor. When the operator binds the
    # builder to the cheap prefix model, its usage is also billed to the cheap
    # role so the strong-side ledger keeps measuring completion only.
    builder_on_cheap_prefix_model = (
        contract_builder_executor == "cheap_prefix_model"
    )
    if builder_on_cheap_prefix_model:
        if not contract_builder_model:
            raise ValueError("contract_builder_model_required")
        if contract_builder_llm_args is None:
            raise ValueError("contract_builder_llm_args_required")
        # Residual closure is the other place contract construction spends a
        # call, and it spends it on the same kind of work: read the whole
        # visible request against a draft the runner already grounded, and
        # return the difference. Nothing in it completes the residual, so which
        # model executes it is an operator choice rather than a method claim,
        # and the role it is billed to should be the role that ran it.
        builder_call_is_request_reading = (
            contract_builder_variant == "model_authored_request_semantics"
            or (
                contract_builder_variant == "deterministic_candidate_grounded"
                and bool(deterministic_contract_closure)
            )
        )
        if not builder_call_is_request_reading:
            raise ValueError("contract_builder_executor_requires_model_authored")
    effective_contract_builder_model = (
        str(contract_builder_model)
        if builder_on_cheap_prefix_model
        else strong_model
    )
    effective_contract_builder_llm_args: Mapping[str, Any] = (
        dict(contract_builder_llm_args or {})
        if builder_on_cheap_prefix_model
        else strong_llm_args
    )
    contract_builder_ledger_role = (
        "cheap" if builder_on_cheap_prefix_model else "strong"
    )

    # Only the candidate-grounded builder is allowed to read the withheld
    # cheap candidate. Both candidate-independent builders derive the contract
    # from the request, the accepted prefix, and receipts alone.
    candidate_bound_contract = (
        contract_builder_variant == "deterministic_candidate_grounded"
    )

    request_semantics_builder = (
        contract_builder_variant == "model_authored_request_semantics"
    )
    # Contract construction is grounding followed by residual closure. The
    # candidate-grounded builder performs grounding without a provider call;
    # closure is the one call that reconciles the grounded obligations with
    # the whole visible request, and it returns only the difference.
    closure_pass_enabled = bool(deterministic_contract_closure) and (
        contract_builder_variant == "deterministic_candidate_grounded"
    )
    effective_builder_attempt_cap = (
        int(contract_builder_attempt_cap)
        if request_semantics_builder
        else (1 if closure_pass_enabled else 0)
    )
    # The zero-obligation fallback re-grounds the same contract once when the
    # candidate committed to nothing, so it can add exactly one builder call
    # on the rows where it fires. Declare it in the cap rather than letting it
    # overrun a budget the artifact reports as smaller.
    if zero_obligation_request_fallback:
        effective_builder_attempt_cap += 1
    total_strong_provider_call_cap = (
        effective_builder_attempt_cap + int(strong_call_cap)
    )

    from tau2.agent.llm_agent import LLMAgent
    from tau2.data_model.message import (
        AssistantMessage,
        MultiToolMessage,
        SystemMessage,
        ToolCall,
        UserMessage,
    )
    from tau2.registry import registry
    from tau2.utils.llm_utils import generate


    class Tau2CFRCAgent(LLMAgent):
        def __init__(self, *agent_args: Any, task: Any = None, **kwargs: Any):
            super().__init__(*agent_args, **kwargs)
            self._task = task
            self._graph_core_strong_used = False
            self._residual_phase_closed = False
            self._pending_suffix_ack = ""
            self._pending_suffix_tool_calls: list[dict[str, Any]] = []
            self._pending_suffix_messages: list[str] = []
            self._pending_suffix_result_public: dict[str, Any] | None = None
            self._committed_commitments: list[Commitment] = []
            self._pending_frontier_tool_names: list[str] = []
            self._pending_frontier_tool_calls: list[dict[str, Any]] = []
            self._pending_frontier_cheap_message: Any | None = None
            self._pending_frontier_pre_write_skipped = False
            self._pending_frontier_residual_activation_reason = ""
            self._pending_frontier_candidate_executed_before_frontier: bool | None = None
            self._pending_frontier_receipt_summary: dict[str, Any] | None = None
            self._request_contract: RequestScopedContract | None = None
            self._request_contract_message: Any | None = None
            self._request_contract_messages: list[Any] = []
            self._request_contract_builder_phases: list[str] = []
            self._request_contract_call_count = 0
            self._request_contract_coverage: ContractCoverage | None = None
            self._strong_calls_used = 0
            self._contract_calls: list[ContractCall] = []
            self._contract_phase_index = 0
            self._ledger_closed = False
            self._request_contract_parse_errors: list[str] = []
            # Why a residual-closure delta was thrown away, counted rather than
            # assigned. The parse-error slot above is overwritten by whichever
            # contract step runs last, so a closure rejection that a later
            # grounding step follows leaves no trace of what it cost us.
            self._closure_rejection_reasons: dict[str, int] = {}
            self._contract_refused_native = False
            self._latest_state: Any | None = None
            self._pending_phase_feedback: list[dict[str, Any]] = []
            self._pending_suffix_plan: list[dict[str, Any]] = []
            self._pending_suffix_inflight: list[dict[str, Any]] = []
            self._replica_receipt_mismatch_latched = False
            self._pending_contract_grounding_plan: list[dict[str, Any]] = []
            self._pending_contract_grounding_inflight: list[dict[str, Any]] = []
            self._contract_grounding_read_signatures: set[str] = set()
            tool_names = [str(tool.name) for tool in self.tools]
            self._mutating_tool_names = set(
                normalize_tool_names(
                    mutating_tool_names
                    if mutating_tool_names is not None
                    else _mutating_tool_names_from_tau_tools(self.tools, tool_names)
                )
            )
            self._support_read_tool_names = {
                name for name in tool_names if name not in self._mutating_tool_names
            }
            # Under the request contract, a terminal handoff call is a
            # commitment frontier too: the cheap model declaring it cannot
            # complete the request is exactly a reached state with the whole
            # remaining request as residual.  The contract then either binds
            # the real write obligations or, when policy demands escalation,
            # the one grounded transfer obligation.  Nine full114 rows ended
            # in an unexamined native give-up transfer under the narrower
            # trigger set.
            self._frontier_trigger_tool_names = set(self._mutating_tool_names)
            self._contract_allowed_tool_names = set(self._mutating_tool_names)
            if request_contract_verifier:
                handoff_names = _HANDOFF_TOOL_NAMES & set(tool_names)
                self._frontier_trigger_tool_names |= handoff_names
                self._contract_allowed_tool_names |= handoff_names

        def _generate_next_message(self, message: Any, state: Any):
            self._latest_state = state
            if isinstance(message, UserMessage) and message.is_audio:
                raise ValueError("User message cannot be audio. Use VoiceLLMAgent instead.")
            if isinstance(message, MultiToolMessage):
                state.messages.extend(message.tool_messages)
            else:
                state.messages.append(message)

            if (
                self._pending_contract_grounding_inflight
                and request_contract_verifier
            ):
                if isinstance(message, MultiToolMessage):
                    grounding_receipts = list(message.tool_messages)
                elif _message_attr_str(message, "role") == "tool":
                    grounding_receipts = [message]
                else:
                    grounding_receipts = []
                return self._handle_contract_grounding_receipts(
                    state,
                    grounding_receipts,
                )

            if self._pending_suffix_inflight and request_contract_verifier:
                if isinstance(message, MultiToolMessage):
                    contract_receipts = list(message.tool_messages)
                elif _message_attr_str(message, "role") == "tool" or _message_attr_str(
                    message, "tool_call_id"
                ):
                    contract_receipts = [message]
                else:
                    contract_receipts = []
                return self._handle_contract_suffix_receipts(
                    state, contract_receipts
                )

            if (
                request_contract_verifier
                and self._residual_phase_closed
                and isinstance(message, UserMessage)
            ):
                return self._run_post_closure_turn(state)

            if self._pending_suffix_tool_calls:
                final_message, receipt_summary = _suffix_final_message_from_receipts(
                    message.tool_messages if isinstance(message, MultiToolMessage) else []
                )
                result_public = dict(self._pending_suffix_result_public or {})
                selected_messages = list(self._pending_suffix_messages)
                self._pending_suffix_tool_calls = []
                self._pending_suffix_messages = []
                self._pending_suffix_ack = ""
                self._pending_suffix_result_public = None
                self._residual_phase_closed = True
                if receipt_summary["all_succeeded"] and selected_messages:
                    content = selected_messages[0]
                    suffix_final_source = "selected_terminal_message"
                else:
                    content = final_message
                    suffix_final_source = "deterministic_receipt_message"
                result_public["fallback_to_cheap_control"] = False
                result_public["fallback_source"] = ""
                result_public["deterministic_suffix_final"] = True
                result_public["suffix_final_source"] = suffix_final_source
                result_public["suffix_receipt_summary"] = receipt_summary
                result_public["receipt_summary"] = receipt_summary
                return AssistantMessage.text(
                    content,
                    raw_data={"papc_cfrc": result_public},
                )

            if self._pending_suffix_ack:
                content = self._pending_suffix_ack
                self._pending_suffix_ack = ""
                return AssistantMessage.text(
                    content,
                    raw_data={"papc_cfrc": {"deterministic_suffix_final": True}},
                )

            if (
                self._pending_frontier_tool_names
                and not self._graph_core_strong_used
            ):
                if isinstance(message, MultiToolMessage):
                    receipt_summary = _tool_receipt_summary(message.tool_messages)
                    self._pending_frontier_receipt_summary = dict(receipt_summary)
                    self._committed_commitments.extend(
                        _receipt_bound_commitments(
                            self._pending_frontier_tool_calls,
                            message.tool_messages,
                        )
                    )
                    if frontier_timing == "pre_write_candidate":
                        if (
                            self._pending_frontier_pre_write_skipped
                            and not receipt_summary["has_error"]
                        ):
                            self._clear_pending_frontier()
                        else:
                            self._pending_frontier_residual_activation_reason = (
                                "post_write_receipt_error_after_pending_carrier_defer"
                                if self._pending_frontier_pre_write_skipped
                                else "pre_write_pending_carrier_after_receipt"
                            )
                            self._pending_frontier_candidate_executed_before_frontier = True
                            return self._run_strong_residual(state)
                    elif (
                        post_write_residual_policy == "receipt_error"
                        and not receipt_summary["has_error"]
                    ):
                        self._clear_pending_frontier()
                    else:
                        self._pending_frontier_residual_activation_reason = (
                            "post_write_receipt_error"
                            if receipt_summary["has_error"]
                            else "post_write_residual_activation_always_after_receipt"
                        )
                        self._pending_frontier_candidate_executed_before_frontier = True
                        return self._run_strong_residual(state)
                else:
                    if (
                        frontier_timing == "pre_write_candidate"
                        and self._pending_frontier_pre_write_skipped
                    ):
                        self._clear_pending_frontier()
                    else:
                        self._pending_frontier_residual_activation_reason = (
                            "pre_write_pending_carrier_without_receipt"
                            if frontier_timing == "pre_write_candidate"
                            else "post_write_pending_frontier_without_receipt"
                        )
                        self._pending_frontier_candidate_executed_before_frontier = (
                            frontier_timing == "post_tool_result"
                        )
                        return self._run_strong_residual(state)

            if post_cfrc_cheap_control_blocked(
                residual_phase_closed=self._residual_phase_closed,
                pending_suffix_ack=self._pending_suffix_ack,
            ):
                return AssistantMessage.text(
                    CFRC_POST_DECISION_TERMINAL_MESSAGE,
                    raw_data={
                        "papc_cfrc": {
                            "residual_phase_closed": True,
                            "reason": "post_cfrc_cheap_control_blocked",
                            "public_safe": True,
                        }
                    },
                )

            cheap_message = self._ensure_model_output_payload(
                self._generate_cheap_prefix(state),
                source="cheap_prefix",
            )
            cheap_tool_calls = _message_tool_calls(cheap_message)
            frontier = [
                call
                for call in cheap_tool_calls
                if call.name in self._frontier_trigger_tool_names
            ]
            if frontier and not self._graph_core_strong_used:
                self._set_pending_frontier(frontier, cheap_message=cheap_message)
                if frontier_timing == "pre_write_candidate":
                    residual_activation_reason = (
                        _pre_write_candidate_residual_activation_reason(
                            frontier,
                            state.messages,
                            policy=pre_write_residual_policy,
                        )
                    )
                    if (
                        residual_activation_reason
                        and self._prior_mutating_tool_call_count(state) > 0
                    ):
                        # The frontier is the first commitment of the
                        # episode.  Once a live state effect exists the
                        # reached state is no longer pre-write, so a later
                        # candidate is ordinary native continuation.  This
                        # also bounds the verifier to one attempt per
                        # episode: after a pre-write abstain releases the
                        # candidate, its write lands and no further
                        # activation follows.
                        residual_activation_reason = None
                    if residual_activation_reason:
                        self._pending_frontier_residual_activation_reason = (
                            residual_activation_reason
                        )
                        self._pending_frontier_candidate_executed_before_frontier = False
                        return self._run_strong_residual(state)
                    self._pending_frontier_pre_write_skipped = True
                    self._pending_frontier_residual_activation_reason = (
                        "pre_write_first_write_candidate_deferred"
                    )
                    return cheap_message
            return cheap_message

        def _set_pending_frontier(self, frontier: Sequence[Any], *, cheap_message: Any) -> None:
            self._pending_frontier_tool_names = [call.name for call in frontier]
            self._pending_frontier_tool_calls = [
                _tool_call_prompt_json(call) for call in frontier
            ]
            self._pending_frontier_cheap_message = cheap_message
            self._pending_frontier_pre_write_skipped = False
            self._pending_frontier_residual_activation_reason = ""
            self._pending_frontier_candidate_executed_before_frontier = None
            self._pending_frontier_receipt_summary = None

        def _clear_pending_frontier(self) -> None:
            self._pending_frontier_tool_names = []
            self._pending_frontier_tool_calls = []
            self._pending_frontier_cheap_message = None
            self._pending_frontier_pre_write_skipped = False
            self._pending_frontier_residual_activation_reason = ""
            self._pending_frontier_candidate_executed_before_frontier = None
            self._pending_frontier_receipt_summary = None

        def _prior_mutating_tool_call_count(self, state: Any) -> int:
            return _visible_mutating_tool_call_count(
                getattr(state, "messages", []),
                self._mutating_tool_names,
            )

        def _verify_request_contract(
            self,
            *,
            state: Any,
            visible_evidence: Sequence[Mapping[str, Any]],
        ) -> RequestScopedContract | Any:
            """Construct the reached-state contract deterministically.

            The withheld candidate is the cheap agent's translation of the
            visible request into exact calls, so the contract instantiates
            one action obligation per withheld write, admissible only when
            every authority-bearing argument value is already visible in
            the request or an accepted receipt.  Receipt arithmetic that
            the request obliges the terminal to report attaches as exact
            literals.  No provider call authors any contract field; policy
            admissibility is judged by executing the graph on the private
            reached-state replica, whose refusals are the environment's
            own policy answers.
            """

            if self._request_contract is not None:
                return self._request_contract
            if request_semantics_builder:
                return self._build_model_authored_request_semantics_contract(
                    state=state,
                    visible_evidence=visible_evidence,
                )
            derived_rows = (
                _retail_request_contract_derived_evidence(
                    messages=state.messages,
                    candidate_calls=self._pending_frontier_tool_calls,
                )
                if domain == "retail"
                else []
            )
            grounding_values = collect_visible_values(
                _all_visible_user_text(state.messages),
                visible_evidence,
                *[
                    str(_message_attr(message, "content") or "")
                    for message in state.messages
                    if _message_attr_str(message, "role") == "tool"
                    or _message_attr_str(message, "tool_call_id")
                ],
            )
            if self._request_contract_call_count >= int(
                max_request_contract_calls
            ):
                return self._return_request_contract_failure(
                    reason="request_contract_call_budget_exhausted",
                    request_contract_messages=[],
                    state=state,
                )
            self._request_contract_call_count += 1
            candidate_names = {
                str(call.get("name") or "").strip()
                for call in self._pending_frontier_tool_calls
            }
            if (
                candidate_names == {"transfer_to_human_agents"}
                and self._prior_mutating_tool_call_count(state) == 0
                and not _visible_user_requests_escalation(state.messages)
            ):
                # A zero-write transfer with no visible escalation request
                # is the cheap agent abandoning the request, not a
                # commitment to keep. The contract freezes the reached
                # state with an open completion: the successor proposes a
                # grounded residual graph, whole-graph admission freezes
                # that proposal into the obligations, and if no admissible
                # graph exists the withheld transfer is released unchanged.
                if zero_obligation_request_fallback:
                    # Contract construction is grounding followed by residual
                    # closure. Here grounding has nothing to ground: the
                    # candidate expressed no commitment, so the obligation
                    # ledger is empty and the residual space it induces is
                    # empty too. Checked Execution can then only abstain, and
                    # the successor's proposal rounds are spent against a
                    # certificate no graph can discharge. The fail-closed
                    # answer is not to release the abandonment; it is to
                    # ground the same contract against the visible request
                    # and the accepted receipts instead of against a
                    # candidate that committed to nothing.
                    return (
                        self._build_model_authored_request_semantics_contract(
                            state=state,
                            visible_evidence=visible_evidence,
                            extra_call_allowance=1,
                            force_single_pass=True,
                        )
                    )
                contract = RequestScopedContract(
                    surface=f"tau2_{domain}",
                    status="ready",
                    communication_requirements=(
                        "complete the visible request from accepted "
                        "evidence, or confirm the transfer",
                    ),
                )
                if closure_pass_enabled:
                    contract = self._close_grounded_contract(
                        draft=contract,
                        state=state,
                        visible_evidence=visible_evidence,
                        grounding_values=grounding_values,
                    )
                self._request_contract = contract
                return contract
            obligations: list[RequestObligation] = []
            grounding_errors: list[str] = []
            for index, call in enumerate(self._pending_frontier_tool_calls, 1):
                tool_name = str(call.get("name") or "").strip()
                arguments = _mapping_or_none(call.get("arguments")) or {}
                if tool_name not in self._contract_allowed_tool_names:
                    grounding_errors.append(
                        "request_contract_tool_not_authorizable"
                    )
                    continue
                try:
                    _assert_argument_values_visible(
                        arguments,
                        visible_values=grounding_values,
                        tool_name=tool_name,
                    )
                except ValueError as exc:
                    grounding_errors.append(str(exc))
                    continue
                obligations.append(
                    RequestObligation(
                        obligation_id=f"candidate_write_{index}",
                        tool_name=tool_name,
                        arguments=arguments,
                        description=(
                            f"withheld candidate write {index}: {tool_name}"
                        ),
                        depends_on=(
                            (f"candidate_write_{index - 1}",)
                            if index > 1
                            else ()
                        ),
                    )
                )
            if grounding_errors or not obligations:
                self._request_contract_parse_errors = sorted(
                    set(grounding_errors)
                ) or ["request_contract_candidate_empty"]
                return self._return_request_contract_failure(
                    reason="request_contract_candidate_ungrounded",
                    request_contract_messages=[],
                    state=state,
                )
            literals = _derived_terminal_literals(
                derived_rows, self._pending_frontier_tool_calls
            )
            contract = RequestScopedContract(
                surface=f"tau2_{domain}",
                status="ready",
                obligations=tuple(obligations),
                required_terminal_literals=tuple(literals),
                all_or_nothing=len(obligations) > 1,
            )
            if closure_pass_enabled:
                contract = self._close_grounded_contract(
                    draft=contract,
                    state=state,
                    visible_evidence=visible_evidence,
                    grounding_values=grounding_values,
                )
            self._request_contract = contract
            return contract

        def _request_contract_receipt_rows(
            self, state: Any
        ) -> list[dict[str, Any]]:
            """Accepted receipts as the contract builder reads them."""

            rows: list[dict[str, Any]] = []
            for receipt_index, message in enumerate(state.messages):
                if (
                    _message_attr_str(message, "role") != "tool"
                    and not _message_attr_str(message, "tool_call_id")
                ):
                    continue
                content = str(_message_attr(message, "content") or "").strip()
                if not content:
                    continue
                rows.append(
                    {
                        "event_id": f"full_receipt_{receipt_index}",
                        "kind": "visible_tool_result_full",
                        "content": _truncate_text(
                            _request_contract_receipt_text(content),
                            2000,
                        ),
                    }
                )
            return rows

        def _close_grounded_contract(
            self,
            *,
            draft: RequestScopedContract,
            state: Any,
            visible_evidence: Sequence[Mapping[str, Any]],
            grounding_values: frozenset[str],
        ) -> RequestScopedContract:
            """Close the grounded ledger against the whole visible request.

            Grounding fixes what the accepted prefix committed to. The request
            can oblige more than the prefix reached, and closure owes exactly
            that difference, so the builder returns the difference rather than
            an authored copy of a draft the runtime already holds. A delta the
            merge or the contract parser rejects leaves the grounded draft
            standing: closure can extend a contract, never destroy one.
            """

            compact_rows = [
                row
                for row in visible_evidence
                if str(row.get("kind") or "") != "visible_tool_result"
            ]
            prompt = request_contract_prompt(
                surface=f"tau2_{domain}",
                visible_request=_all_visible_user_text(state.messages),
                visible_evidence=[
                    *compact_rows,
                    *self._request_contract_receipt_rows(state),
                ],
                accepted_prefix_context=(
                    _untrusted_accepted_prefix_context(state.messages)
                ),
                candidate_calls=[],
                public_tool_schemas=_public_tau_tool_schemas(
                    [
                        tool
                        for tool in self.tools
                        if str(getattr(tool, "name", "") or "")
                        in self._contract_allowed_tool_names
                    ]
                ),
                public_policy=str(getattr(self, "domain_policy", "") or ""),
            )
            closure_prompt = {
                **prompt,
                "canonical_grounded_draft": draft.private_json(),
                "output_schema": closure_delta_output_schema(),
                "rules": [*prompt["rules"], *closure_delta_rules()],
                "construction_phase": {
                    "schema": "papc.tau2.contract_builder_phase.v1",
                    "phase": "deterministic_residual_closure",
                    "instruction": (
                        "Re-check the entire visible request and every accepted "
                        "receipt against canonical_grounded_draft, then return "
                        "only the residual difference."
                    ),
                },
            }
            closure_message = generate(
                model=effective_contract_builder_model,
                messages=[
                    SystemMessage(
                        role="system",
                        content=_GROUNDED_RESIDUAL_CLOSURE_SYSTEM_PROMPT,
                    ),
                    UserMessage.text(
                        json.dumps(
                            closure_prompt,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    ),
                ],
                call_name=(
                    f"{domain}_request_contract_deterministic_residual_closure"
                ),
                **dict(effective_contract_builder_llm_args),
            )
            self._request_contract_message = closure_message
            self._request_contract_messages.append(closure_message)
            try:
                contract = parse_request_contract(
                    apply_closure_delta(
                        draft=draft,
                        delta=getattr(closure_message, "content", "") or "",
                    ),
                    surface=f"tau2_{domain}",
                    allowed_tool_names=sorted(
                        self._contract_allowed_tool_names
                    ),
                    visible_values=grounding_values,
                    visible_literal_text=self._terminal_literal_carrier(
                        state=state,
                        visible_evidence=visible_evidence,
                    ),
                )
            except (TypeError, ValueError) as exc:
                self._request_contract_parse_errors = [
                    f"deterministic_residual_closure:{exc}"
                ]
                self._record_closure_rejection(str(exc).split(":", 1)[0])
                self._request_contract_builder_phases.append(
                    "deterministic_residual_closure_rejected"
                )
                return draft
            if contract.status != "ready":
                # Closure reports a difference; it does not decide that the
                # grounded commitment is unservable. A non-ready delta leaves
                # the draft standing rather than dissolving obligations the
                # accepted prefix already made.
                self._request_contract_parse_errors = [
                    "deterministic_residual_closure:"
                    f"non_ready_status:{contract.status}"
                ]
                self._record_closure_rejection(
                    f"non_ready_status:{contract.status}"
                )
                self._request_contract_builder_phases.append(
                    "deterministic_residual_closure_rejected"
                )
                return draft
            self._request_contract_builder_phases.append(
                "deterministic_residual_closure"
            )
            return contract

        def _record_closure_rejection(self, reason: str) -> None:
            key = str(reason or "unknown").strip() or "unknown"
            self._closure_rejection_reasons[key] = (
                int(self._closure_rejection_reasons.get(key, 0)) + 1
            )

        def _terminal_literal_carrier(
            self,
            *,
            state: Any,
            visible_evidence: Sequence[Mapping[str, Any]],
        ) -> str:
            """Carrier text a builder-authored terminal literal must appear in.

            Empty means the parser keeps every authored literal, which is the
            registered behaviour."""

            if not grounded_terminal_literals:
                return ""
            return _grounded_literal_carrier_text(
                state.messages,
                derived_evidence=visible_evidence,
                public_policy=str(getattr(self, "domain_policy", "") or ""),
            )

        def _build_model_authored_request_semantics_contract(
            self,
            *,
            state: Any,
            visible_evidence: Sequence[Mapping[str, Any]],
            extra_call_allowance: int = 0,
            force_single_pass: bool = False,
        ) -> RequestScopedContract | Any:
            """Build request-level O from request semantics and accepted receipts."""

            receipt_rows = self._request_contract_receipt_rows(state)
            # Do not target-condition request semantics on a cheap action.
            # Contract-bound arithmetic is derived only after O is frozen.
            derived_rows: list[dict[str, Any]] = []
            compact_rows = [
                row
                for row in visible_evidence
                if str(row.get("kind") or "") != "visible_tool_result"
            ]
            prompt = request_contract_prompt(
                surface=f"tau2_{domain}",
                visible_request=_all_visible_user_text(state.messages),
                visible_evidence=[
                    *compact_rows,
                    *receipt_rows,
                    *derived_rows,
                ],
                accepted_prefix_context=(
                    _untrusted_accepted_prefix_context(state.messages)
                ),
                # The model-authored builder receives no cheap action
                # candidate, even as a hypothesis: request semantics are
                # constructed only from the request, accepted receipts,
                # public schemas, and policy.
                candidate_calls=[],
                public_tool_schemas=_public_tau_tool_schemas(
                    [
                        tool
                        for tool in self.tools
                        if str(getattr(tool, "name", "") or "")
                        in self._contract_allowed_tool_names
                    ]
                ),
                public_policy=str(getattr(self, "domain_policy", "") or ""),
            )
            grounding_values = collect_visible_values(
                _all_visible_user_text(state.messages),
                visible_evidence,
                *[
                    str(_message_attr(message, "content") or "")
                    for message in state.messages
                    if _message_attr_str(message, "role") == "tool"
                    or _message_attr_str(message, "tool_call_id")
                ],
            )
            if self._request_contract_call_count >= (
                int(max_request_contract_calls) + int(extra_call_allowance)
            ):
                return self._return_request_contract_failure(
                    reason="request_contract_call_budget_exhausted",
                    request_contract_messages=[],
                    state=state,
                )
            self._request_contract_call_count += 1
            builder_messages: list[Any] = []
            if force_single_pass or effective_builder_attempt_cap == 1:
                closure_prompt = {
                    **prompt,
                    "construction_phase": {
                        "schema": "papc.tau2.contract_builder_phase.v1",
                        "phase": "request_semantics_closure",
                        "instruction": (
                            "Construct and close the complete request-level "
                            "obligation ledger in one pass. Re-check the entire "
                            "visible request and every accepted receipt before "
                            "returning it; include every still-valid requested "
                            "state change and communication obligation."
                        ),
                    },
                }
                closure_message = generate(
                    model=effective_contract_builder_model,
                    messages=[
                        SystemMessage(
                            role="system",
                            content=(
                                _MODEL_AUTHORED_REQUEST_CONTRACT_SINGLE_PASS_SYSTEM_PROMPT
                            ),
                        ),
                        UserMessage.text(
                            json.dumps(
                                closure_prompt,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                        ),
                    ],
                    call_name=(
                        f"{domain}_request_contract_builder_residual_closure"
                    ),
                    **dict(effective_contract_builder_llm_args),
                )
                builder_messages.append(closure_message)
                self._request_contract_message = closure_message
                self._request_contract_messages.append(closure_message)
                self._request_contract_builder_phases.append(
                    "request_semantics_closure"
                )
                try:
                    contract = parse_request_contract(
                        getattr(closure_message, "content", "") or "",
                        surface=f"tau2_{domain}",
                        allowed_tool_names=sorted(
                            self._contract_allowed_tool_names
                        ),
                        visible_values=grounding_values,
                        visible_literal_text=self._terminal_literal_carrier(
                            state=state,
                            visible_evidence=visible_evidence,
                        ),
                    )
                except (TypeError, ValueError) as exc:
                    self._request_contract_parse_errors = [
                        f"request_semantics_closure:{exc}"
                    ]
                    return self._return_request_contract_failure(
                        reason="request_contract_closure_parse_failed",
                        request_contract_messages=builder_messages,
                        state=state,
                    )
                contract = replace(contract, authored_by="model_authored")
                self._request_contract = contract
                return contract

            draft_prompt = {
                **prompt,
                "construction_phase": {
                    "schema": "papc.tau2.contract_builder_phase.v1",
                    "phase": "request_semantics_draft",
                    "instruction": (
                        "Draft the complete request-level obligation ledger from "
                        "the visible request and accepted receipts."
                    ),
                },
            }
            draft_message = generate(
                model=effective_contract_builder_model,
                messages=[
                    SystemMessage(
                        role="system",
                        content=_MODEL_AUTHORED_REQUEST_CONTRACT_SYSTEM_PROMPT,
                    ),
                    UserMessage.text(
                        json.dumps(
                            draft_prompt,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    ),
                ],
                call_name=f"{domain}_request_contract_builder_draft",
                **dict(effective_contract_builder_llm_args),
            )
            builder_messages.append(draft_message)
            self._request_contract_message = draft_message
            self._request_contract_messages.append(draft_message)
            self._request_contract_builder_phases.append(
                "request_semantics_draft"
            )
            try:
                draft_contract = parse_request_contract(
                    getattr(draft_message, "content", "") or "",
                    surface=f"tau2_{domain}",
                    allowed_tool_names=sorted(
                        self._contract_allowed_tool_names
                    ),
                    visible_values=grounding_values,
                    visible_literal_text=self._terminal_literal_carrier(
                        state=state,
                        visible_evidence=visible_evidence,
                    ),
                )
            except (TypeError, ValueError) as exc:
                self._request_contract_parse_errors = [
                    f"request_semantics_draft:{exc}"
                ]
                return self._return_request_contract_failure(
                    reason="request_contract_draft_parse_failed",
                    request_contract_messages=builder_messages,
                    state=state,
                )

            closure_prompt = {
                **prompt,
                "canonical_parsed_draft": draft_contract.private_json(),
                "construction_phase": {
                    "schema": "papc.tau2.contract_builder_phase.v1",
                    "phase": "residual_closure_revision",
                    "instruction": (
                        "Re-check the entire visible request and every accepted "
                        "receipt against canonical_parsed_draft. Return one revised "
                        "final contract that closes every still-valid requested "
                        "state change and communication obligation; do not merely "
                        "approve the draft."
                    ),
                },
            }
            closure_message = generate(
                model=effective_contract_builder_model,
                messages=[
                    SystemMessage(
                        role="system",
                        content=(
                            _MODEL_AUTHORED_REQUEST_CONTRACT_CLOSURE_SYSTEM_PROMPT
                        ),
                    ),
                    UserMessage.text(
                        json.dumps(
                            closure_prompt,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    ),
                ],
                call_name=f"{domain}_request_contract_builder_residual_closure",
                **dict(effective_contract_builder_llm_args),
            )
            builder_messages.append(closure_message)
            self._request_contract_message = closure_message
            self._request_contract_messages.append(closure_message)
            self._request_contract_builder_phases.append(
                "residual_closure_revision"
            )
            try:
                contract = parse_request_contract(
                    getattr(closure_message, "content", "") or "",
                    surface=f"tau2_{domain}",
                    allowed_tool_names=sorted(
                        self._contract_allowed_tool_names
                    ),
                    visible_values=grounding_values,
                    visible_literal_text=self._terminal_literal_carrier(
                        state=state,
                        visible_evidence=visible_evidence,
                    ),
                )
            except (TypeError, ValueError) as exc:
                self._request_contract_parse_errors = [
                    f"residual_closure_revision:{exc}"
                ]
                return self._return_request_contract_failure(
                    reason="request_contract_closure_parse_failed",
                    request_contract_messages=builder_messages,
                    state=state,
                )
            contract = replace(contract, authored_by="model_authored")
            self._request_contract = contract
            return contract

        def _return_request_contract_failure(
            self,
            *,
            reason: str,
            request_contract_messages: Sequence[Any],
            state: Any | None = None,
        ):
            # Candidate transcription preserves the frozen failure return:
            # before any live write, release the candidate it used to author
            # the contract.  A request-semantics builder cannot do that because
            # the candidate is explicitly untrusted and may be the reason
            # construction failed; that optional configuration stays closed.
            if (
                contract_builder_variant == "deterministic_candidate_grounded"
                and state is not None
                and not self._contract_calls
            ):
                return self._defer_to_native_candidate(
                    state,
                    reason=reason,
                    result_public={},
                    candidate_message=self._pending_frontier_cheap_message,
                    candidate_executed=bool(
                        self._pending_frontier_candidate_executed_before_frontier
                    ),
                    strong_messages=request_contract_messages,
                )
            certificate = build_residual_certificate(
                domain=domain,
                allowed_residual_tools=self._contract_allowed_tool_names,
                commitments=self._committed_commitments,
                residual_obligations=(),
                evidence_facts=(),
                frontier_tool_names=(
                    self._pending_frontier_tool_names
                    if candidate_bound_contract
                    else ()
                ),
                evidence_event_count=0,
            )
            proposal = CFRCProposalGraph(abstain_reason=reason)
            result = _abstain(
                certificate, proposal, reason=reason,
                projection={
                    "decision": "project_abstain",
                    "request_contract_verifier": True,
                    "request_contract_parse_error_codes": list(
                        self._request_contract_parse_errors
                    ),
                },
            )
            _write_agent_decision_row(
                contract_builder_ledger_role=contract_builder_ledger_role,
                path=decision_log_path,
                task=getattr(self._task, "id", ""),
                frontier_timing=frontier_timing,
                cheap_sidecar_message=self._pending_frontier_cheap_message,
                strong_messages=request_contract_messages,
                result=result,
                strong_call_cap_configured=int(strong_call_cap),
                contract_builder_variant=contract_builder_variant,
                contract_builder_attempt_cap_configured=(
                    effective_builder_attempt_cap
                ),
                contract_builder_phase_names=(
                    self._request_contract_builder_phases[
                        : len(request_contract_messages)
                    ]
                ),
                closure_rejection_reasons=dict(self._closure_rejection_reasons),
                total_strong_provider_call_cap_configured=(
                    total_strong_provider_call_cap
                ),
                frontier_candidate_private=(
                    list(self._pending_frontier_tool_calls)
                    if frontier_timing == "pre_write_candidate"
                    else []
                ),
                residual_activation_reason=(
                    self._pending_frontier_residual_activation_reason
                ),
                candidate_tool_call_executed_before_frontier=(
                    self._pending_frontier_candidate_executed_before_frontier
                ),
                receipt_summary=self._pending_frontier_receipt_summary,
                graph_residual_continuation={
                    "schema": "papc.tau2.graph_residual_continuation.v1",
                    "enabled": False,
                    "policy": "request_contract_failure_return",
                    "after_residual_suffix": False,
                    "after_abstain": False,
                    "layer": "graph_residual_continuation",
                    "strict_core_counted": False,
                },
            )
            self._clear_pending_frontier()
            self._residual_phase_closed = True
            # ledger_closed keeps its actual obligation truth: this refusal
            # path is only reachable after method writes exist, so forcing it
            # would fabricate a closed ledger.
            self._request_contract = None
            return AssistantMessage.text(
                "I could not verify the complete retail request, so I stopped "
                "without making an additional change.",
                raw_data={
                    "papc_cfrc": {
                        **result.public_json(),
                        "hand_back_to_cheap_control": False,
                        "fail_closed_before_next_live_write": True,
                        "request_contract_verifier": True,
                        "checked_execution_completion": False,
                    }
                },
            )

        def _return_model_authored_contract_without_suffix(
            self,
            state: Any,
            *,
            contract: RequestScopedContract,
            request_contract_messages: Sequence[Any],
        ):
            """Honor a non-ready model-authored contract without candidate release."""

            reason = f"request_contract_{contract.status}"
            certificate = build_residual_certificate(
                domain=domain,
                allowed_residual_tools=self._contract_allowed_tool_names,
                commitments=self._committed_commitments,
                residual_obligations=(),
                evidence_facts=(),
                frontier_tool_names=(
                    self._pending_frontier_tool_names
                    if candidate_bound_contract
                    else ()
                ),
                evidence_event_count=0,
                message_residual_allowed=True,
            )
            result = _abstain(
                certificate,
                CFRCProposalGraph(abstain_reason=reason),
                reason=reason,
                projection={
                    "decision": "project_abstain",
                    "request_contract": contract.public_json(),
                },
            )
            if contract.status != "no_state_change" and not self._contract_calls:
                # No contract could be established, so there is no residual to
                # complete and nothing has been realized under method
                # authority. That failure value ends the method's authority,
                # not the episode: the withheld native candidate continues on
                # the authority it always had, exactly as it does when an
                # admitted proposal fails before the first live write.
                return self._defer_to_native_candidate(
                    state,
                    reason=reason,
                    result_public=result.public_json(),
                    candidate_message=self._pending_frontier_cheap_message,
                    candidate_executed=bool(
                        self._pending_frontier_candidate_executed_before_frontier
                    ),
                    strong_messages=request_contract_messages,
                )
            _write_agent_decision_row(
                contract_builder_ledger_role=contract_builder_ledger_role,
                path=decision_log_path,
                task=getattr(self._task, "id", ""),
                frontier_timing=frontier_timing,
                cheap_sidecar_message=self._pending_frontier_cheap_message,
                strong_messages=request_contract_messages,
                result=result,
                strong_call_cap_configured=int(strong_call_cap),
                contract_builder_variant=contract_builder_variant,
                contract_builder_attempt_cap_configured=(
                    effective_builder_attempt_cap
                ),
                contract_builder_phase_names=(
                    self._request_contract_builder_phases[
                        : len(request_contract_messages)
                    ]
                ),
                closure_rejection_reasons=dict(self._closure_rejection_reasons),
                total_strong_provider_call_cap_configured=(
                    total_strong_provider_call_cap
                ),
                frontier_candidate_private=(
                    list(self._pending_frontier_tool_calls)
                    if frontier_timing == "pre_write_candidate"
                    else []
                ),
                residual_activation_reason=(
                    self._pending_frontier_residual_activation_reason
                ),
                candidate_tool_call_executed_before_frontier=(
                    self._pending_frontier_candidate_executed_before_frontier
                ),
                receipt_summary=self._pending_frontier_receipt_summary,
                graph_residual_continuation={
                    "schema": "papc.tau2.graph_residual_continuation.v1",
                    "enabled": False,
                    "policy": "model_authored_contract_failure_return",
                    "after_residual_suffix": False,
                    "after_abstain": False,
                    "layer": "graph_residual_continuation",
                    "strict_core_counted": False,
                },
            )
            self._clear_pending_frontier()
            if contract.status == "no_state_change":
                self._residual_phase_closed = True
                self._ledger_closed = True
                return AssistantMessage.text(
                    contract.terminal_response,
                    raw_data={
                        "papc_cfrc": {
                            **result.public_json(),
                            **self._contract_public_state(),
                            "request_contract_no_state_change": True,
                            "hand_back_to_cheap_control": False,
                            "fail_closed_before_next_live_write": True,
                            "checked_execution_completion": False,
                        }
                    },
                )
            return self._contract_fail_closed_terminal(
                reason=reason,
                result_public=result.public_json(),
            )

        def _run_strong_residual(self, state: Any):
            self._graph_core_strong_used = True
            fallback_cheap_message = self._pending_frontier_cheap_message
            fallback_candidate_executed = bool(
                self._pending_frontier_candidate_executed_before_frontier
            )
            visible_evidence = compact_visible_evidence(
                state.messages, max_events=24
            )
            retail_derived_evidence: list[dict[str, Any]] = []
            request_contract = None
            contract_event_id = "request_contract"
            effective_suffix_tool_call_budget = int(suffix_tool_call_budget)
            effective_message_residual = bool(allow_message_residual)
            request_contract_messages: list[Any] = []
            if request_contract_verifier:
                # Observing the reached state is component one's own job, and a
                # read changes nothing, so it neither moves the frontier nor
                # spends a provider call. Without it the successor can only
                # cite values the accepted prefix happened to have fetched,
                # and any argument that needs one more read is unreachable.
                grounding_plan = (
                    self._request_contract_grounding_plan(state=state)
                    if candidate_bound_contract
                    or effective_builder_attempt_cap == 1
                    else []
                )
                if grounding_plan:
                    self._pending_contract_grounding_plan = grounding_plan
                    return self._emit_next_contract_grounding_call()
            if request_contract_verifier:
                request_contract_was_frozen = self._request_contract is not None
                request_contract = self._verify_request_contract(
                    state=state,
                    visible_evidence=visible_evidence,
                )
                if isinstance(request_contract, AssistantMessage):
                    return request_contract
                request_contract_messages = (
                    []
                    if request_contract_was_frozen
                    else list(self._request_contract_messages)
                )
                if (
                    request_contract.authored_by == "model_authored"
                    and request_contract.status != "ready"
                ):
                    return self._return_model_authored_contract_without_suffix(
                        state,
                        contract=request_contract,
                        request_contract_messages=request_contract_messages,
                    )
                if len(request_contract.obligations) > int(
                    request_contract_suffix_cap
                ):
                    return self._return_request_contract_failure(
                        reason="request_contract_suffix_cap_exceeded",
                        request_contract_messages=request_contract_messages,
                        state=state,
                    )
                if domain == "retail":
                    arithmetic_targets = (
                        [
                            {
                                "name": obligation.tool_name,
                                "arguments": dict(obligation.arguments),
                            }
                            for obligation in request_contract.obligations
                        ]
                        if request_semantics_builder or closure_pass_enabled
                        else list(self._pending_frontier_tool_calls)
                    )
                    retail_derived_evidence = (
                        _retail_request_contract_derived_evidence(
                            messages=state.messages,
                            candidate_calls=arithmetic_targets,
                            target_source=(
                                "frozen_request_contract"
                                if request_semantics_builder
                                or closure_pass_enabled
                                else "untrusted_candidate"
                            ),
                        )
                    )
                effective_suffix_tool_call_budget = max(
                    effective_suffix_tool_call_budget,
                    len(request_contract.obligations),
                )
                effective_message_residual = bool(
                    allow_message_residual
                    or request_contract.communication_requirements
                    or request_contract.required_terminal_literals
                )
                visible_evidence = [
                    *visible_evidence,
                    *retail_derived_evidence,
                    {
                        "event_id": contract_event_id,
                        "kind": "request_scoped_obligation_contract",
                        "content": request_contract.private_json(),
                    },
                ]
                residual_obligations = [
                    ResidualObligation(
                        obligation_id=item.obligation_id,
                        required_unit_tokens=(
                            specific_tool_effect_unit(
                                item.tool_name, item.arguments
                            ),
                        ),
                        effect_families=(tool_effect_unit(item.tool_name),),
                        provenance_refs=(contract_event_id,),
                        full_discharge_required=True,
                    )
                    for item in request_contract.obligations
                ]
                if effective_message_residual and (
                    not request_contract_private_replica
                    or (
                        request_semantics_builder
                        and not request_contract.obligations
                    )
                ):
                    residual_obligations.append(
                        ResidualObligation(
                            obligation_id="request-contract:terminal-message",
                            required_unit_tokens=(MESSAGE_RESIDUAL_UNIT,),
                            effect_families=(MESSAGE_RESIDUAL_UNIT,),
                            provenance_refs=(contract_event_id,),
                            full_discharge_required=True,
                        )
                    )
                residual_obligations = _open_residual_obligations_after_commitments(
                    residual_obligations, self._committed_commitments
                )
            else:
                residual_obligations = _open_residual_obligations_after_commitments(
                    _frontier_candidate_residual_obligations(
                        self._pending_frontier_tool_calls
                    ),
                    self._committed_commitments,
                )
            ell3_metadata = _schema_derived_ell3_restoration_metadata(
                tools=self.tools,
                commitments=self._committed_commitments,
                compiler_setting=compiler_setting,
            )
            evidence_facts = _evidence_facts_from_visible_rows(visible_evidence)
            certificate = build_residual_certificate(
                domain=domain,
                allowed_residual_tools=self._contract_allowed_tool_names,
                commitments=self._committed_commitments,
                residual_obligations=residual_obligations,
                evidence_facts=evidence_facts,
                frontier_tool_names=(
                    self._pending_frontier_tool_names
                    if candidate_bound_contract
                    else ()
                ),
                evidence_event_count=len(state.messages),
                message_residual_allowed=effective_message_residual,
                collision_budget=ell3_metadata["collision_budget"],
                restoration_cost=ell3_metadata["restoration_cost"],
            )
            _assert_no_explicit_k_r_overlap(certificate)
            completion_contract_open = bool(
                request_contract is not None
                and not request_contract.obligations
                and contract_builder_variant
                == "deterministic_candidate_grounded"
            )
            if not completion_contract_open and not strong_call_required(
                certificate
            ):
                proposal = CFRCProposalGraph(
                    abstain_reason="no_verifier_issued_open_residual_obligation"
                )
                projection_start = time.perf_counter()
                result = project_cfrc_proposal(
                    parameter_grounding_texts=_parameter_grounding_texts(self.tools),
                    certificate=certificate,
                    proposal=proposal,
                    available_tool_names=[str(tool.name) for tool in self.tools],
                    required_carrier_tool_calls=(
                        self._pending_frontier_tool_calls
                        if frontier_timing == "pre_write_candidate"
                        and request_contract is None
                        else []
                    ),
                    visible_evidence=visible_evidence,
                    compiler_setting=compiler_setting,
                    compiler_suffix_tool_call_budget=suffix_tool_call_budget,
                )
                projection_seconds = time.perf_counter() - projection_start
                continuation_policy = _graph_residual_continuation_policy(
                    result=result,
                    has_tool_suffix=False,
                    fallback_after_tool_suffix_receipt=False,
                    fallback_after_abstain=True,
                )
                _write_agent_decision_row(
                contract_builder_ledger_role=contract_builder_ledger_role,
                    path=decision_log_path,
                    task=getattr(self._task, "id", ""),
                    frontier_timing=frontier_timing,
                    frontier_candidate_private=(
                        list(self._pending_frontier_tool_calls)
                        if frontier_timing == "pre_write_candidate"
                        else []
                    ),
                    cheap_sidecar_message=(
                        self._pending_frontier_cheap_message
                        if frontier_timing == "pre_write_candidate"
                        else None
                    ),
                    strong_messages=request_contract_messages,
                    result=result,
                    strong_rounds=[],
                    strong_call_cap_configured=int(strong_call_cap),
                    contract_builder_variant=contract_builder_variant,
                    contract_builder_attempt_cap_configured=(
                        effective_builder_attempt_cap
                    ),
                    contract_builder_phase_names=(
                        self._request_contract_builder_phases[
                            : len(request_contract_messages)
                        ]
                    ),
                    closure_rejection_reasons=dict(
                        self._closure_rejection_reasons
                    ),
                    total_strong_provider_call_cap_configured=(
                        total_strong_provider_call_cap
                    ),
                    projection_seconds=projection_seconds,
                    residual_activation_reason=(
                        self._pending_frontier_residual_activation_reason
                    ),
                    candidate_tool_call_executed_before_frontier=(
                        self._pending_frontier_candidate_executed_before_frontier
                    ),
                    receipt_summary=self._pending_frontier_receipt_summary,
                    graph_residual_continuation=continuation_policy,
                )
                self._residual_phase_closed = False
                self._clear_pending_frontier()
                if request_contract_verifier:
                    self._residual_phase_closed = True
                    self._ledger_closed = True
                    content = (
                        request_contract.terminal_response
                        if request_contract is not None
                        and request_contract.terminal_response
                        else (
                            "Every requested update is already reflected in the "
                            "current state, so I made no additional change."
                        )
                    )
                    return AssistantMessage.text(
                        content,
                        raw_data={
                            "papc_cfrc": {
                                **result.public_json(),
                                **self._contract_public_state(),
                                "hand_back_to_cheap_control": False,
                                "fail_closed_before_next_live_write": True,
                                "request_contract_verifier": True,
                            }
                        },
                    )
                if (
                    frontier_timing == "pre_write_candidate"
                    and fallback_cheap_message is not None
                ):
                    return self._ensure_model_output_payload(
                        _attach_cfrc_fallback_metadata(
                            fallback_cheap_message,
                            result,
                            fallback_source="pre_write_candidate_original_cheap_message",
                        ),
                        source="pre_write_candidate_original_cheap_message",
                    )
                cheap_fallback_message = self._generate_cheap_prefix(state)
                return self._ensure_model_output_payload(
                    _attach_cfrc_fallback_metadata(
                        cheap_fallback_message,
                        result,
                        fallback_source="post_tool_result_cheap_continuation",
                    ),
                    source="post_tool_result_cheap_continuation",
                )
            prompt_payload = build_strong_prompt_payload(
                certificate=certificate,
                visible_evidence=visible_evidence,
                residual_tools=_residual_tool_schemas(
                    self.tools, self._mutating_tool_names
                ),
                support_read_tools=_support_read_tool_schemas(
                    self.tools, self._support_read_tool_names
                ),
                cheap_frontier_tool_calls=(
                    list(self._pending_frontier_tool_calls)
                    if include_frontier_candidate_hint
                    and contract_builder_variant
                    == "deterministic_candidate_grounded"
                    else []
                ),
                frontier_candidate_is_verified_carrier=(
                    frontier_timing == "pre_write_candidate"
                    and contract_builder_variant
                    == "deterministic_candidate_grounded"
                ),
                suffix_tool_call_budget=effective_suffix_tool_call_budget,
            )
            prompt_payload["initial_visible_user_request"] = _all_visible_user_text(
                state.messages
            )
            if request_contract is not None:
                prompt_payload["request_scoped_obligation_contract"] = (
                    contract_feedback(request_contract)
                )
                prompt_payload["request_contract_is_frozen"] = True
                if (
                    not request_contract.obligations
                    and contract_builder_variant
                    == "deterministic_candidate_grounded"
                ):
                    prompt_payload["open_completion_contract"] = {
                        "schema": "papc.tau2.open_completion_contract.v1",
                        "instruction": (
                            "The withheld candidate is a give-up transfer "
                            "with zero prior writes and no visible user "
                            "escalation request. Propose the complete "
                            "residual graph that resolves the visible "
                            "request: every write argument value must be "
                            "copied verbatim from the visible request or an "
                            "accepted receipt, or the graph is rejected. If "
                            "no admissible write resolves the request, "
                            "return an abstain proposal and the transfer "
                            "proceeds unchanged."
                        ),
                    }
            visible_evidence = list(prompt_payload["visible_evidence"])
            deterministic_first_proposal = None
            if (
                request_contract is not None
                and request_contract.obligations
                and frontier_timing == "pre_write_candidate"
                and not self._contract_calls
                and (
                    contract_builder_variant
                    == "deterministic_candidate_grounded"
                    or effective_builder_attempt_cap == 1
                )
            ):
                deterministic_first_proposal = (
                    _contract_transcription_proposal_json(
                        contract=request_contract,
                        candidate_calls=self._pending_frontier_tool_calls,
                        include_terminal_message=bool(
                            effective_message_residual
                        ),
                    )
                )
            freeze_completion = None
            if (
                request_contract is not None
                and not request_contract.obligations
                and contract_builder_variant
                == "deterministic_candidate_grounded"
            ):
                completion_grounding_values = collect_visible_values(
                    _all_visible_user_text(state.messages),
                    visible_evidence,
                    *[
                        str(_message_attr(message, "content") or "")
                        for message in state.messages
                        if _message_attr_str(message, "role") == "tool"
                        or _message_attr_str(message, "tool_call_id")
                    ],
                )
                completion_surface = request_contract.surface
                completion_state_messages = list(state.messages)

                def freeze_completion(proposal_nodes):
                    """Freeze an admissible completion proposal into the
                    contract: one obligation per grounded proposed write,
                    in proposal order. Returns (contract, certificate) or
                    None when any write is unauthorized or ungrounded."""

                    admitted: list[RequestObligation] = []
                    previous_id = ""
                    for node in proposal_nodes:
                        if node.kind != "tool_call":
                            continue
                        if (
                            node.tool_name
                            not in self._contract_allowed_tool_names
                            or node.tool_name == "transfer_to_human_agents"
                        ):
                            return None
                        try:
                            _assert_argument_values_visible(
                                node.arguments,
                                visible_values=completion_grounding_values,
                                tool_name=node.tool_name,
                            )
                        except ValueError:
                            return None
                        obligation_id = f"admitted_write_{len(admitted) + 1}"
                        admitted.append(
                            RequestObligation(
                                obligation_id=obligation_id,
                                tool_name=node.tool_name,
                                arguments=node.arguments,
                                description=(
                                    "successor completion admitted at "
                                    "whole-graph admission"
                                ),
                                depends_on=(
                                    (previous_id,) if previous_id else ()
                                ),
                            )
                        )
                        previous_id = obligation_id
                    if not admitted:
                        return None
                    frozen = RequestScopedContract(
                        surface=completion_surface,
                        status="ready",
                        obligations=tuple(admitted),
                        all_or_nothing=len(admitted) > 1,
                    )
                    frozen_certificate = build_residual_certificate(
                        domain=domain,
                        allowed_residual_tools=(
                            self._contract_allowed_tool_names
                        ),
                        commitments=self._committed_commitments,
                        residual_obligations=[
                            ResidualObligation(
                                obligation_id=item.obligation_id,
                                required_unit_tokens=(
                                    specific_tool_effect_unit(
                                        item.tool_name, item.arguments
                                    ),
                                ),
                                effect_families=(
                                    tool_effect_unit(item.tool_name),
                                ),
                                provenance_refs=(contract_event_id,),
                                full_discharge_required=True,
                            )
                            for item in frozen.obligations
                        ],
                        evidence_facts=_evidence_facts_from_visible_rows(
                            visible_evidence
                        ),
                        frontier_tool_names=(
                            self._pending_frontier_tool_names
                            if candidate_bound_contract
                            else ()
                        ),
                        evidence_event_count=len(
                            completion_state_messages
                        ),
                        message_residual_allowed=(
                            effective_message_residual
                        ),
                    )
                    self._request_contract = frozen
                    contract_row = {
                        "event_id": contract_event_id,
                        "kind": "request_scoped_obligation_contract",
                        "content": frozen.private_json(),
                    }
                    return frozen, frozen_certificate, contract_row

            rounds = self._run_strong_proposal_rounds(
                prompt_payload=prompt_payload,
                certificate=certificate,
                available_tool_names=[str(tool.name) for tool in self.tools],
                visible_evidence=visible_evidence,
                request_contract=request_contract,
                effective_suffix_tool_call_budget=(
                    effective_suffix_tool_call_budget
                ),
                deterministic_first_proposal=deterministic_first_proposal,
                freeze_completion=freeze_completion,
            )
            if not rounds:
                if not self._contract_calls:
                    return self._defer_to_native_candidate(
                        state,
                        reason="strong_call_budget_exhausted",
                        result_public={},
                        candidate_message=fallback_cheap_message,
                        candidate_executed=fallback_candidate_executed,
                        strong_messages=request_contract_messages,
                    )
                self._clear_pending_frontier()
                return self._contract_fail_closed_terminal(
                    reason="strong_call_budget_exhausted"
                )
            selected_round = _select_compiler_constrained_round(rounds)
            result = selected_round["result"]
            projection_seconds = sum(
                float(round_row["projection_seconds"]) for round_row in rounds
            )
            suffix_calls = [
                ToolCall(
                    id=f"papc_cfrc_{index}",
                    name=call["name"],
                    arguments=dict(call["arguments"]),
                )
                for index, call in enumerate(result.suffix_tool_calls())
            ]
            continuation_policy = _graph_residual_continuation_policy(
                result=result,
                has_tool_suffix=bool(suffix_calls),
                fallback_after_tool_suffix_receipt=False,
                fallback_after_abstain=not request_contract_verifier,
            )
            _write_agent_decision_row(
                contract_builder_ledger_role=contract_builder_ledger_role,
                path=decision_log_path,
                task=getattr(self._task, "id", ""),
                frontier_timing=frontier_timing,
                frontier_candidate_private=(
                    list(self._pending_frontier_tool_calls)
                    if frontier_timing == "pre_write_candidate"
                    else []
                ),
                cheap_sidecar_message=(
                    self._pending_frontier_cheap_message
                    if frontier_timing == "pre_write_candidate"
                    else None
                ),
                strong_messages=[
                    *request_contract_messages,
                    *[
                        round_row["strong_message"]
                        for round_row in rounds
                        if not round_row.get("deterministic_transcription")
                    ],
                ],
                result=result,
                strong_rounds=rounds,
                strong_call_cap_configured=int(strong_call_cap),
                contract_builder_variant=contract_builder_variant,
                contract_builder_attempt_cap_configured=(
                    effective_builder_attempt_cap
                ),
                contract_builder_phase_names=(
                    self._request_contract_builder_phases[
                        : len(request_contract_messages)
                    ]
                ),
                closure_rejection_reasons=dict(self._closure_rejection_reasons),
                total_strong_provider_call_cap_configured=(
                    total_strong_provider_call_cap
                ),
                projection_seconds=projection_seconds,
                residual_activation_reason=(
                    self._pending_frontier_residual_activation_reason
                ),
                candidate_tool_call_executed_before_frontier=(
                    self._pending_frontier_candidate_executed_before_frontier
                ),
                receipt_summary=self._pending_frontier_receipt_summary,
                graph_residual_continuation=continuation_policy,
            )
            self._residual_phase_closed = False
            self._clear_pending_frontier()
            if request_contract_verifier and result.execute:
                contract_suffix_messages = result.suffix_messages()
                deterministic_terminal_text = str(
                    selected_round.get("deterministic_terminal_text") or ""
                ).strip()
                if deterministic_terminal_text:
                    contract_suffix_messages = [deterministic_terminal_text]
                contract_result_public = result.public_json()
                if deterministic_terminal_text:
                    contract_result_public[
                        "deterministic_terminal_literal_completion"
                    ] = True
                if result.suffix_tool_calls():
                    round_witnesses = selected_round.get(
                        "replica_witnesses", {}
                    )
                    if not isinstance(round_witnesses, Mapping):
                        round_witnesses = {}
                    self._pending_suffix_plan = []
                    tool_nodes = [
                        node
                        for node in result.selected_nodes
                        if node.kind == "tool_call"
                    ]
                    for index, node in enumerate(tool_nodes):
                        witness = round_witnesses.get(node.node_id, ("", False))
                        expected_receipt_ref = (
                            str(witness[0])
                            if isinstance(witness, tuple) and witness
                            else ""
                        )
                        self._pending_suffix_plan.append({
                            "tool_call_id": (
                                f"papc_cfrc_p{self._contract_phase_index}_{index}"
                            ),
                            "graph_node_id": node.node_id,
                            "name": node.tool_name,
                            "arguments": dict(node.arguments),
                            "expected_receipt_ref": expected_receipt_ref,
                        })
                    self._pending_suffix_messages = contract_suffix_messages
                    self._pending_suffix_result_public = contract_result_public
                    return self._emit_next_contract_call()
                if contract_suffix_messages:
                    return self._finalize_contract_scope(
                        state,
                        terminal_text=contract_suffix_messages[0],
                        result_public=contract_result_public,
                    )
            if result.execute and suffix_calls:
                self._pending_suffix_tool_calls = list(result.suffix_tool_calls())
                self._pending_suffix_messages = result.suffix_messages()
                self._pending_suffix_result_public = result.public_json()
                return AssistantMessage(
                    role="assistant",
                    content=None,
                    tool_calls=suffix_calls,
                    raw_data={"papc_cfrc": result.public_json()},
                )
            suffix_messages = result.suffix_messages()
            if result.execute and suffix_messages:
                return AssistantMessage.text(
                    suffix_messages[0],
                    raw_data={"papc_cfrc": result.public_json()},
                )
            if request_contract_verifier:
                if not self._contract_calls:
                    # No live contract write has executed, so this is the
                    # formal pre-write failure return: the method abstains
                    # without Checked Execution and control returns to the
                    # native candidate, mirroring the contract-verification
                    # abstains.  Replica witnesses are proposal-time only
                    # and release no live effect.  After the first live
                    # contract write the failure latch below stays closed.
                    # The condition is the realized-effect state, not which
                    # builder produced the contract: the method's failure
                    # value ends its own authority, never the incumbent's.
                    return self._defer_to_native_candidate(
                        state,
                        reason=str(
                            result.reason or "residual_proposal_not_admitted"
                        ),
                        result_public=result.public_json(),
                        candidate_message=fallback_cheap_message,
                        candidate_executed=fallback_candidate_executed,
                        # The proposal decision row immediately above owns
                        # every builder/proposal call made in this phase.
                        # This continuation row records only the control
                        # transition, so its provider-call delta is empty.
                        strong_messages=[],
                    )
                return self._contract_fail_closed_terminal(
                    reason=str(result.reason or "residual_proposal_not_admitted"),
                    result_public=result.public_json(),
                )
            if (
                frontier_timing == "pre_write_candidate"
                and fallback_cheap_message is not None
            ):
                return self._ensure_model_output_payload(
                    _attach_cfrc_fallback_metadata(
                        fallback_cheap_message,
                        result,
                        fallback_source="pre_write_candidate_original_cheap_message",
                    ),
                    source="pre_write_candidate_original_cheap_message",
                )
            cheap_fallback_message = self._generate_cheap_prefix(state)
            return self._ensure_model_output_payload(
                _attach_cfrc_fallback_metadata(
                    cheap_fallback_message,
                    result,
                    fallback_source="post_tool_result_cheap_continuation",
                ),
                source="post_tool_result_cheap_continuation",
            )

        def _defer_to_native_candidate(
            self,
            state: Any,
            *,
            reason: str,
            result_public: Mapping[str, Any],
            candidate_message: Any | None,
            candidate_executed: bool,
            strong_messages: Sequence[Any] = (),
        ):
            """Formal pre-write failure return with native continuation.

            Entering here requires that no live contract write has executed:
            the method abstains before Checked Execution and the withheld
            native candidate (or a regenerated cheap turn) continues, with
            the decision row recording the abstain and continuation mode.
            After the first live contract write the failure latch keeps the
            episode fail-closed and never reaches this helper.
            """

            continuation_mode = (
                "verbatim_candidate"
                if (
                    frontier_timing == "pre_write_candidate"
                    and candidate_message is not None
                    and not candidate_executed
                )
                else "regenerated_cheap_turn"
            )
            certificate = build_residual_certificate(
                domain=domain,
                allowed_residual_tools=self._contract_allowed_tool_names,
                commitments=self._committed_commitments,
                residual_obligations=(),
                evidence_facts=(),
                frontier_tool_names=(
                    self._pending_frontier_tool_names
                    if candidate_bound_contract
                    else ()
                ),
                evidence_event_count=len(getattr(state, "messages", []) or []),
                message_residual_allowed=True,
            )
            abstain_result = _abstain(
                certificate,
                CFRCProposalGraph(abstain_reason=reason),
                reason=reason,
                projection={
                    "decision": "cfrc_pre_write_abstain_native_continuation",
                    "request_contract_verifier": True,
                    "request_contract_parse_error_codes": list(
                        self._request_contract_parse_errors
                    ),
                },
            )
            _write_agent_decision_row(
                contract_builder_ledger_role=contract_builder_ledger_role,
                path=decision_log_path,
                task=getattr(self._task, "id", ""),
                frontier_timing=frontier_timing,
                cheap_sidecar_message=self._pending_frontier_cheap_message,
                strong_messages=strong_messages,
                result=abstain_result,
                strong_call_cap_configured=int(strong_call_cap),
                contract_builder_variant=contract_builder_variant,
                contract_builder_attempt_cap_configured=(
                    effective_builder_attempt_cap
                ),
                contract_builder_phase_names=(
                    self._request_contract_builder_phases[: len(strong_messages)]
                ),
                closure_rejection_reasons=dict(self._closure_rejection_reasons),
                total_strong_provider_call_cap_configured=(
                    total_strong_provider_call_cap
                ),
                frontier_candidate_private=(
                    list(self._pending_frontier_tool_calls)
                    if frontier_timing == "pre_write_candidate"
                    else []
                ),
                residual_activation_reason=(
                    self._pending_frontier_residual_activation_reason
                ),
                candidate_tool_call_executed_before_frontier=candidate_executed,
                receipt_summary=self._pending_frontier_receipt_summary,
                graph_residual_continuation={
                    "schema": "papc.tau2.graph_residual_continuation.v1",
                    "enabled": True,
                    "policy": "native_candidate_after_pre_write_abstain",
                    "after_residual_suffix": False,
                    "after_abstain": True,
                    "layer": "graph_residual_continuation",
                    "strict_core_counted": False,
                    "native_continuation_mode": continuation_mode,
                },
            )
            self._clear_pending_frontier()
            # A release ends this construction unit; a later candidate opens
            # a fresh unit that must consume the construction budget even
            # when this unit had grounding reads.
            self._contract_grounding_read_signatures.clear()
            self._contract_refused_native = True
            abstain_meta = {
                **dict(result_public),
                **self._contract_public_state(),
                "cfrc_abstained_pre_write": True,
                "native_continuation_after_abstain": True,
                "checked_execution_completion": False,
                "abstain_reason_code": reason,
                "native_continuation_mode": continuation_mode,
                "public_safe": True,
            }
            if continuation_mode == "verbatim_candidate":
                return self._ensure_model_output_payload(
                    _attach_cfrc_metadata(
                        _released_candidate_copy(candidate_message), abstain_meta
                    ),
                    source="cfrc_pre_write_abstain_candidate",
                )
            cheap_message = self._generate_cheap_prefix(state)
            return self._ensure_model_output_payload(
                _attach_cfrc_metadata(cheap_message, abstain_meta),
                source="cfrc_pre_write_abstain_cheap_continuation",
            )

        def _request_contract_grounding_plan(
            self,
            *,
            state: Any,
        ) -> list[dict[str, Any]]:
            """Resolve finite collection and variant evidence before contract freeze."""

            remaining_budget = max(
                0,
                int(request_contract_suffix_cap)
                - len(self._contract_grounding_read_signatures),
            )
            if remaining_budget == 0:
                return []
            if candidate_bound_contract:
                candidate_order_ids = {
                    str(
                        (call.get("arguments") or {}).get("order_id") or ""
                    ).strip()
                    for call in self._pending_frontier_tool_calls
                    if isinstance(call.get("arguments"), Mapping)
                }
                candidate_order_ids.discard("")
                expand_account_orders = bool(candidate_order_ids) and not all(
                    _identifier_first_introduced_by_user(
                        getattr(state, "messages", []),
                        order_id,
                    )
                    for order_id in candidate_order_ids
                )
            else:
                # A candidate-independent contract observes the account the
                # accepted prefix already authenticated, bounded by the same
                # read budget. Nothing here reads the withheld candidate, and
                # the rule is the same for every request.
                candidate_order_ids = set()
                expand_account_orders = True
            order_ids: list[str] = []
            inspected_order_ids: set[str] = set()
            order_receipts: dict[str, Mapping[str, Any]] = {}
            inspected_product_ids: set[str] = set()
            for message in getattr(state, "messages", []) or []:
                if (
                    _message_attr_str(message, "role") != "tool"
                    and not _message_attr_str(message, "tool_call_id")
                ):
                    continue
                payload = _receipt_content_value(
                    _message_attr(message, "content")
                )
                if not isinstance(payload, Mapping):
                    continue
                raw_orders = payload.get("orders")
                if isinstance(raw_orders, list):
                    for raw_order_id in raw_orders:
                        order_id = str(raw_order_id or "").strip()
                        if order_id and order_id not in order_ids:
                            order_ids.append(order_id)
                inspected_order_id = str(payload.get("order_id") or "").strip()
                if inspected_order_id:
                    inspected_order_ids.add(inspected_order_id)
                    order_receipts[inspected_order_id] = payload
                inspected_product_id = str(
                    payload.get("product_id") or ""
                ).strip()
                if inspected_product_id and isinstance(
                    payload.get("variants"), Mapping
                ):
                    inspected_product_ids.add(inspected_product_id)
            missing_order_ids = [
                order_id
                for order_id in order_ids
                if order_id not in inspected_order_ids
            ] if expand_account_orders else []
            calls: list[dict[str, Any]] = []
            if (
                "get_order_details" in self._support_read_tool_names
                and missing_order_ids
                and len(missing_order_ids) <= remaining_budget
            ):
                calls.extend(
                    {
                        "name": "get_order_details",
                        "arguments": {"order_id": order_id},
                    }
                    for order_id in missing_order_ids
                )

            # A target item receipt proves availability and price but does
            # not prove that the target belongs to the same product as the
            # current order item.  Resolve that link with the public product
            # read identified by the exact current item in the exact order.
            # This is read-only contract grounding; it grants no authority to
            # the withheld write candidate.
            if (
                not calls
                and "get_product_details" in self._support_read_tool_names
            ):
                product_ids: list[str] = []
                for call in self._pending_frontier_tool_calls:
                    arguments = _mapping_or_none(call.get("arguments")) or {}
                    order_id = str(arguments.get("order_id") or "").strip()
                    current_item_ids = [
                        str(item_id or "").strip()
                        for item_id in _list_value(arguments.get("item_ids"))
                        if str(item_id or "").strip()
                    ]
                    new_item_ids = [
                        str(item_id or "").strip()
                        for item_id in _list_value(arguments.get("new_item_ids"))
                        if str(item_id or "").strip()
                    ]
                    receipt = order_receipts.get(order_id)
                    receipt_items = (
                        receipt.get("items")
                        if isinstance(receipt, Mapping)
                        else None
                    )
                    if (
                        not current_item_ids
                        or not new_item_ids
                        or len(current_item_ids) != len(new_item_ids)
                        or not isinstance(receipt_items, list)
                    ):
                        continue
                    product_by_item = {
                        str(item.get("item_id") or "").strip(): str(
                            item.get("product_id") or ""
                        ).strip()
                        for item in receipt_items
                        if isinstance(item, Mapping)
                    }
                    for current_item_id in current_item_ids:
                        product_id = product_by_item.get(current_item_id, "")
                        if (
                            product_id
                            and product_id not in inspected_product_ids
                            and product_id not in product_ids
                        ):
                            product_ids.append(product_id)
                if product_ids and len(product_ids) <= remaining_budget:
                    calls.extend(
                        {
                            "name": "get_product_details",
                            "arguments": {"product_id": product_id},
                        }
                        for product_id in product_ids
                    )

            planned: list[dict[str, Any]] = []
            for call in calls:
                signature = _tool_call_signature(
                    str(call["name"]),
                    _mapping_or_none(call["arguments"]) or {},
                )
                if signature in self._contract_grounding_read_signatures:
                    continue
                planned.append(call)
                self._contract_grounding_read_signatures.add(signature)
            start_index = (
                len(self._contract_grounding_read_signatures) - len(planned)
            )
            return [
                {
                    "tool_call_id": (
                        f"papc_cfrc_grounding_{start_index + index}"
                    ),
                    **call,
                }
                for index, call in enumerate(planned)
            ]

        def _emit_next_contract_grounding_call(self):
            if not self._pending_contract_grounding_plan:
                return self._run_strong_residual(self._latest_state)
            call = self._pending_contract_grounding_plan.pop(0)
            self._pending_contract_grounding_inflight = [call]
            return AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id=str(call["tool_call_id"]),
                        name=str(call["name"]),
                        arguments=dict(call["arguments"]),
                    )
                ],
                raw_data={
                    "papc_cfrc": {
                        "request_contract_grounding_read": True,
                        "remaining_grounding_read_count": len(
                            self._pending_contract_grounding_plan
                        ),
                        "hand_back_to_cheap_control": False,
                        "public_safe": True,
                    }
                },
            )

        def _handle_contract_grounding_receipts(
            self,
            state: Any,
            tool_messages: Sequence[Any],
        ):
            inflight = list(self._pending_contract_grounding_inflight)
            self._pending_contract_grounding_inflight = []
            receipt_ids = {
                _message_attr_str(receipt, "tool_call_id")
                or _message_attr_str(receipt, "id")
                for receipt in tool_messages
            }
            expected_ids = {
                str(call["tool_call_id"]) for call in inflight
            }
            if (
                not inflight
                or receipt_ids != expected_ids
                or any(
                    not _tool_receipt_status(receipt)["success"]
                    for receipt in tool_messages
                )
            ):
                self._pending_contract_grounding_plan = []
                return self._contract_fail_closed_terminal(
                    reason="request_contract_grounding_read_failed"
                )
            if self._pending_contract_grounding_plan:
                return self._emit_next_contract_grounding_call()
            return self._run_strong_residual(state)

        def _proposal_replica_witnesses(
            self,
            selected_nodes: Sequence[CFRCProposalNode],
        ) -> tuple[dict[str, tuple[str, bool]], str]:
            """Execute the exact selected suffix on a private reached-state copy."""

            self._last_replica_failure: dict[str, Any] = {}
            tool_nodes = [
                node
                for node in selected_nodes
                if node.kind == "tool_call"
            ]
            if not tool_nodes:
                return {}, "private_replica_tool_suffix_empty"
            tools_by_name = {
                str(getattr(tool, "name", "") or ""): tool for tool in self.tools
            }
            if any(node.tool_name not in tools_by_name for node in tool_nodes):
                return {}, "private_replica_tool_unavailable"
            owners: list[Any] = []
            for tool in tools_by_name.values():
                owner = getattr(getattr(tool, "_func", None), "__self__", None)
                if (
                    owner is None
                    or not hasattr(owner, REPLICA_PROPOSAL_STATE_ATTRIBUTE)
                    or any(owner is seen for seen in owners)
                ):
                    continue
                owners.append(owner)
            if not owners:
                return {}, "private_reached_state_replica_unavailable"

            live_states = tuple(
                getattr(owner, REPLICA_PROPOSAL_STATE_ATTRIBUTE)
                for owner in owners
            )
            replica_states = deepcopy(live_states)
            witnesses: dict[str, tuple[str, bool]] = {}
            try:
                for owner, replica_state in zip(
                    owners, replica_states, strict=True
                ):
                    setattr(
                        owner,
                        REPLICA_PROPOSAL_STATE_ATTRIBUTE,
                        replica_state,
                    )
                for position, node in enumerate(tool_nodes):
                    succeeded = True
                    try:
                        result_value = json_safe_value(
                            tools_by_name[node.tool_name](**dict(node.arguments))
                        )
                    except Exception as exc:
                        succeeded = False
                        # Which node stopped the suffix and what kind of error
                        # stopped it. Both are structural: a position in the
                        # proposal the successor itself authored, and an
                        # exception class name. Neither names a tool, an
                        # argument, or a corrected value, so the verdict can
                        # reach the next round without telling the successor
                        # what to write instead.
                        self._last_replica_failure = {
                            "failed_node_position": int(position),
                            "tool_node_count": len(tool_nodes),
                            "error_type": type(exc).__name__,
                        }
                        result_value = {
                            "error_type": type(exc).__name__,
                            "replica_execution_failed": True,
                        }
                    witnesses[node.node_id] = (
                        receipt_ref(
                            tool_name=node.tool_name,
                            arguments=node.arguments,
                            result=result_value,
                            instance_id=node.node_id,
                        ),
                        succeeded,
                    )
                    if not succeeded:
                        break
            finally:
                for owner, live_state in zip(
                    owners, live_states, strict=True
                ):
                    setattr(
                        owner,
                        REPLICA_PROPOSAL_STATE_ATTRIBUTE,
                        live_state,
                    )
            if len(witnesses) != len(tool_nodes) or any(
                not succeeded for _, succeeded in witnesses.values()
            ):
                return witnesses, "private_replica_suffix_execution_failed"
            return witnesses, ""

        def _emit_next_contract_call(self):
            succeeded_signatures = {
                call.signature
                for call in self._contract_calls
                if call.succeeded
            }
            while self._pending_suffix_plan:
                head = self._pending_suffix_plan[0]
                signature = ContractCall(
                    node_id=str(head["tool_call_id"]),
                    tool_name=str(head["name"]),
                    arguments=dict(head["arguments"]),
                ).signature
                if signature in succeeded_signatures:
                    self._pending_suffix_plan.pop(0)
                    continue
                break
            if not self._pending_suffix_plan:
                terminal_text = (
                    self._pending_suffix_messages[0]
                    if self._pending_suffix_messages
                    else ""
                )
                result_public = dict(self._pending_suffix_result_public or {})
                return self._finalize_contract_scope(
                    self._latest_state,
                    terminal_text=terminal_text,
                    result_public=result_public,
                )
            head = self._pending_suffix_plan.pop(0)
            if (
                request_contract_private_replica
                and not head.get("expected_receipt_ref")
            ):
                self._pending_suffix_plan = []
                return self._contract_fail_closed_terminal(
                    reason="proposal_replica_receipt_missing",
                    result_public=self._pending_suffix_result_public,
                )
            self._pending_suffix_inflight = [head]
            result_public = dict(self._pending_suffix_result_public or {})
            return AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id=str(head["tool_call_id"]),
                        name=str(head["name"]),
                        arguments=dict(head["arguments"]),
                    )
                ],
                raw_data={
                    "papc_cfrc": {
                        **result_public,
                        **self._contract_public_state(),
                        "sequential_contract_execution": True,
                        "remaining_planned_call_count": len(
                            self._pending_suffix_plan
                        ),
                    }
                },
            )

        def _handle_contract_suffix_receipts(
            self, state: Any, tool_messages: Sequence[Any]
        ):
            inflight = list(self._pending_suffix_inflight)
            self._pending_suffix_inflight = []
            receipts_by_id = {
                receipt_id: receipt
                for receipt in tool_messages
                if (
                    receipt_id := (
                        _message_attr_str(receipt, "tool_call_id")
                        or _message_attr_str(receipt, "id")
                    )
                )
            }
            phase_failed = False
            for call in inflight:
                receipt = receipts_by_id.get(str(call["tool_call_id"]))
                succeeded = bool(
                    receipt is not None
                    and _tool_receipt_status(receipt)["success"]
                )
                if (
                    request_contract_private_replica
                    and call.get("expected_receipt_ref")
                ):
                    actual_receipt_ref = (
                        receipt_ref(
                            tool_name=str(call["name"]),
                            arguments=dict(call["arguments"]),
                            result=_receipt_content_value(
                                _message_attr(receipt, "content")
                            ),
                            instance_id=str(call.get("graph_node_id") or ""),
                        )
                        if receipt is not None
                        else ""
                    )
                    if (
                        not succeeded
                        or actual_receipt_ref
                        != str(call["expected_receipt_ref"])
                    ):
                        succeeded = False
                        self._replica_receipt_mismatch_latched = True
                self._contract_calls.append(
                    ContractCall(
                        node_id=str(call["tool_call_id"]),
                        tool_name=str(call["name"]),
                        arguments=dict(call["arguments"]),
                        succeeded=succeeded,
                    )
                )
                if not succeeded:
                    phase_failed = True
            self._committed_commitments.extend(
                _receipt_bound_commitments(inflight, tool_messages)
            )
            if self._replica_receipt_mismatch_latched:
                self._pending_suffix_plan = []
                return self._contract_fail_closed_terminal(
                    reason="live_replica_receipt_mismatch",
                    result_public=self._pending_suffix_result_public,
                )
            if not phase_failed and self._pending_suffix_plan:
                return self._emit_next_contract_call()
            terminal_text = (
                self._pending_suffix_messages[0]
                if self._pending_suffix_messages
                else ""
            )
            result_public = dict(self._pending_suffix_result_public or {})
            self._pending_suffix_plan = []
            return self._finalize_contract_scope(
                state,
                terminal_text=terminal_text,
                result_public=result_public,
            )

        def _finalize_contract_scope(
            self,
            state: Any,
            *,
            terminal_text: str,
            result_public: Mapping[str, Any],
        ):
            contract = self._request_contract
            if contract is None:
                return self._contract_fail_closed_terminal(
                    reason="request_contract_missing_at_finalize",
                    result_public=result_public,
                )
            coverage = match_contract_calls(
                contract=contract,
                calls=self._contract_calls,
                write_tool_names=sorted(self._contract_allowed_tool_names),
                terminal_text=terminal_text,
            )
            deterministic_literal_completion = False
            if (
                not coverage.complete
                and not coverage.missing_obligation_ids
                and not coverage.extra_write_node_ids
                and not coverage.dependency_errors
                and coverage.missing_terminal_literals
                and _literals_visibly_grounded(
                    coverage.missing_terminal_literals,
                    getattr(state, "messages", []) or [],
                    public_policy=str(
                        getattr(self, "domain_policy", "") or ""
                    ),
                )
            ):
                # Every write obligation is discharged by an exact receipt
                # and the only open item is the message residual: contract
                # literals the proposal's terminal failed to carry.
                # Carrying them is a deterministic completion of the
                # message residual, but only for literals already present
                # in the visible conversation, because the parser does not
                # ground required_terminal_literals; failing closed here
                # would refuse a request whose state changes all
                # succeeded.
                literal_sentence = _terminal_literal_sentence(
                    coverage.missing_terminal_literals
                )
                base_terminal = (
                    terminal_text
                    or contract.terminal_response
                    or (
                        "The requested updates are done; let me know if "
                        "there is anything else."
                    )
                ).rstrip()
                terminal_text = f"{base_terminal} {literal_sentence}"
                coverage = match_contract_calls(
                    contract=contract,
                    calls=self._contract_calls,
                    write_tool_names=sorted(self._contract_allowed_tool_names),
                    terminal_text=terminal_text,
                )
                deterministic_literal_completion = coverage.complete
            self._request_contract_coverage = coverage
            if coverage.complete:
                self._residual_phase_closed = True
                self._ledger_closed = True
                self._pending_suffix_messages = []
                self._pending_suffix_result_public = None
                if (
                    request_semantics_builder
                ):
                    # The model-authored contract is the complete authority
                    # boundary.  Do not hand a ledger-closed scope back to
                    # the cheap model: a new tool call on that turn would be
                    # outside the frozen contract and could re-release the
                    # withheld candidate.  The terminal below has already
                    # passed exact contract coverage, including every
                    # required_terminal_literal.
                    content = terminal_text.strip() or (
                        contract.terminal_response.strip()
                    )
                    if not content:
                        content = (
                            "The request-scoped contract is complete; let me "
                            "know if there is anything else."
                        )
                    return AssistantMessage.text(
                        content,
                        raw_data={
                            "papc_cfrc": {
                                **dict(result_public),
                                **self._contract_public_state(),
                                "deterministic_suffix_final": not bool(
                                    terminal_text.strip()
                                ),
                                "suffix_final_source": (
                                    "model_authored_contract_terminal"
                                ),
                                "deterministic_terminal_literal_completion": (
                                    deterministic_literal_completion
                                    or bool(
                                        result_public.get(
                                            "deterministic_terminal_literal_completion"
                                        )
                                    )
                                ),
                                "hand_back_to_cheap_control": False,
                            }
                        },
                    )
                # The ledger is closed by matching live receipts, so the
                # residual is discharged; the conversation itself is not
                # over, and a canned completion announcement here reads as
                # a dialogue-ending act that cuts off whatever the user
                # would have said next.  The native agent, which sees the
                # successful receipts in its own transcript, speaks the
                # closing turn; the exact receipt-arithmetic literals the
                # contract obliges attach mechanically to that turn.
                closing = self._generate_cheap_prefix(state)
                closing_calls = _message_tool_calls(closing)
                if closing_calls:
                    return self._ensure_model_output_payload(
                        _attach_cfrc_metadata(
                            closing,
                            {
                                **self._contract_public_state(),
                                "suffix_final_source": (
                                    "ledger_closed_native_turn"
                                ),
                                "hand_back_to_cheap_control": False,
                                "public_safe": True,
                            },
                        ),
                        source="ledger_closed_native_turn",
                    )
                closing_text = str(
                    getattr(closing, "content", "") or ""
                ).strip()
                literal_sentence = ""
                if contract.required_terminal_literals:
                    literal_sentence = _terminal_literal_sentence(
                        contract.required_terminal_literals
                    )
                if not closing_text:
                    closing_text = (
                        "The requested updates are done; let me know if "
                        "there is anything else."
                    )
                content = (
                    f"{closing_text.rstrip()} {literal_sentence}".rstrip()
                    if literal_sentence
                    else closing_text
                )
                return AssistantMessage.text(
                    content,
                    raw_data={
                        "papc_cfrc": {
                            **dict(result_public),
                            **self._contract_public_state(),
                            "deterministic_suffix_final": False,
                            "suffix_final_source": "ledger_closed_terminal",
                            "deterministic_terminal_literal_completion": (
                                deterministic_literal_completion
                                or bool(
                                    result_public.get(
                                        "deterministic_terminal_literal_completion"
                                    )
                                )
                            ),
                            "hand_back_to_cheap_control": False,
                        }
                    },
                )
            if self._strong_calls_used < max(1, int(strong_call_cap)):
                return self._run_followup_contract_phase(state, coverage)
            return self._contract_fail_closed_terminal(
                reason="obligations_open_after_strong_budget",
                result_public=result_public,
            )

        def _run_followup_contract_phase(
            self, state: Any, coverage: ContractCoverage
        ):
            contract = self._request_contract
            self._contract_phase_index += 1
            self._pending_suffix_messages = []
            self._pending_suffix_result_public = None
            feedback: list[dict[str, Any]] = []
            if contract is not None:
                feedback.append(contract_feedback(contract, coverage))
            feedback.append(
                {
                    "schema": "papc.tau2.contract_phase_receipts.v1",
                    "instruction": (
                        "Some contract obligations remain open after live "
                        "receipts. Propose a residual that completes only the "
                        "still-open obligations from the frozen contract. Do "
                        "not repeat calls that already succeeded."
                    ),
                    "executed_calls": [
                        {
                            "tool_name": call.tool_name,
                            "succeeded": call.succeeded,
                        }
                        for call in self._contract_calls
                    ],
                }
            )
            self._pending_phase_feedback = feedback
            return self._run_strong_residual(state)

        def _run_post_closure_turn(self, state: Any):
            if not self._ledger_closed:
                return AssistantMessage.text(
                    CFRC_POST_DECISION_TERMINAL_MESSAGE,
                    raw_data={
                        "papc_cfrc": {
                            **self._contract_public_state(),
                            "reason": "contract_scope_failure_latched",
                            "public_safe": True,
                        }
                    },
                )
            if self._request_contract_call_count >= int(max_request_contract_calls):
                # The residual of this reached state is complete and its
                # ledger is closed, so the method has discharged the single
                # commitment frontier this setting defines.  Whatever the
                # user asks next is a fresh request the native agent owns;
                # holding it behind a refusal loses rows and claims an
                # authority the frontier never granted.
                self._residual_phase_closed = False
                self._ledger_closed = False
                return self._defer_to_native_candidate(
                    state,
                    reason="residual_complete_native_continuation",
                    result_public={},
                    candidate_message=None,
                    candidate_executed=False,
                )
            self._request_contract = None
            self._request_contract_message = None
            self._request_contract_messages = []
            self._request_contract_coverage = None
            # The prior scope's successful calls are now commitments in K.
            # They must not be rematched as candidate calls for a later,
            # request-scoped contract, where they would appear as extra
            # writes and make every otherwise-complete suffix fail closed.
            self._contract_calls = []
            self._pending_phase_feedback = []
            self._contract_grounding_read_signatures = set()
            self._residual_phase_closed = False
            self._ledger_closed = False
            self._contract_phase_index += 1
            self._pending_frontier_residual_activation_reason = (
                "post_closure_new_scope"
            )
            return self._run_strong_residual(state)

        def _contract_public_state(self) -> dict[str, Any]:
            contract = self._request_contract
            coverage = self._request_contract_coverage
            return {
                "schema": "papc.tau2.request_contract_ledger_state.v1",
                "request_contract_verifier": True,
                "contract_present": contract is not None,
                "contract_sha256": (
                    contract.sha256 if contract is not None else ""
                ),
                "contract_obligation_count": (
                    len(contract.obligations) if contract is not None else 0
                ),
                "ledger_closed": bool(self._ledger_closed),
                "residual_phase_closed": bool(self._residual_phase_closed),
                "contract_phase_index": int(self._contract_phase_index),
                "strong_calls_used": int(self._strong_calls_used),
                "executed_contract_call_count": len(self._contract_calls),
                "succeeded_contract_call_count": sum(
                    1 for call in self._contract_calls if call.succeeded
                ),
                "coverage": (
                    coverage.public_json() if coverage is not None else None
                ),
                "public_safe": True,
            }

        def _contract_fail_closed_terminal(
            self,
            *,
            reason: str,
            result_public: Mapping[str, Any] | None = None,
        ):
            self._pending_suffix_plan = []
            self._pending_suffix_inflight = []
            self._pending_suffix_messages = []
            self._pending_suffix_result_public = None
            self._residual_phase_closed = True
            self._ledger_closed = False
            contract = self._request_contract
            coverage = self._request_contract_coverage
            total = len(contract.obligations) if contract is not None else 0
            done = (
                len(coverage.covered_by_obligation)
                if coverage is not None
                else 0
            )
            if total and done:
                content = (
                    f"I completed {done} of {total} requested updates with "
                    "verified receipts, but I could not safely verify the "
                    "remaining updates, so I stopped rather than guess."
                )
            else:
                content = (
                    "I could not safely execute every requested retail action, "
                    "so I stopped without handing the residual back."
                )
            return AssistantMessage.text(
                content,
                raw_data={
                    "papc_cfrc": {
                        **dict(result_public or {}),
                        **self._contract_public_state(),
                        "fail_closed_reason": reason,
                        "hand_back_to_cheap_control": False,
                        "fail_closed_before_next_live_write": True,
                        "request_contract_verifier": True,
                        "checked_execution_completion": False,
                    }
                },
            )

        def _generate_cheap_prefix(self, state: Any) -> Any:
            message = generate(
                model=self.llm,
                tools=self.tools,
                messages=state.system_messages + state.messages,
                call_name="cfrc_cheap_prefix",
                **self.llm_args,
            )
            if _assistant_message_has_payload(message):
                return message
            return generate(
                model=self.llm,
                tools=self.tools,
                messages=state.system_messages + state.messages,
                call_name="cfrc_cheap_prefix_empty_retry",
                **self.llm_args,
            )

        def _ensure_model_output_payload(self, message: Any, *, source: str) -> Any:
            if _assistant_message_has_payload(message):
                return message
            papc_cfrc = _public_cfrc_metadata(message)
            papc_cfrc["empty_model_output_terminal"] = True
            papc_cfrc["model_output_guard"] = {
                "schema": "papc.tau2.model_output_guard.v1",
                "reason": "empty_model_output_terminal",
                "source": source,
                "public_safe": True,
            }
            papc_cfrc["public_safe"] = True
            return AssistantMessage.text(
                CFRC_EMPTY_MODEL_OUTPUT_TERMINAL_MESSAGE,
                raw_data={"papc_cfrc": papc_cfrc},
            )

        def _run_strong_proposal_rounds(
            self,
            *,
            prompt_payload: dict[str, Any],
            certificate: CFRCResidualCertificate,
            available_tool_names: Sequence[str],
            visible_evidence: Sequence[Mapping[str, Any]],
            request_contract: RequestScopedContract | None = None,
            effective_suffix_tool_call_budget: int | None = None,
            deterministic_first_proposal: str | None = None,
            freeze_completion: Any = None,
        ) -> list[dict[str, Any]]:
            rounds: list[dict[str, Any]] = []
            typed_feedback: list[dict[str, Any]] = list(self._pending_phase_feedback)
            self._pending_phase_feedback = []
            call_cap = max(1, int(strong_call_cap))
            if (
                request_contract is not None
                and self._strong_calls_used >= call_cap
                and deterministic_first_proposal is None
            ):
                # The deterministic transcription is not a provider call, so
                # a spent strong budget still admits it; only paid proposal
                # rounds stay closed.
                return rounds
            while True:
                round_index = (
                    self._strong_calls_used + 1
                    if request_contract is not None
                    else len(rounds) + 1
                )
                deterministic_unused = (
                    deterministic_first_proposal is not None
                    and not any(
                        row["deterministic_transcription"] for row in rounds
                    )
                )
                deterministic_round = deterministic_unused and (
                    # Fallback ordering keeps the transcription as the safety
                    # net the successor falls back to: the contract restates
                    # the withheld candidate, so replaying it first is what
                    # denies Executable Residual Completion any say.
                    self._strong_calls_used >= call_cap
                    if transcription_is_fallback
                    else not rounds
                )
                if deterministic_round:
                    # The contract transcription is not a provider call and
                    # spends no strong budget; if its projection does not
                    # execute, the ordinary proposal rounds still run.
                    strong_message = AssistantMessage.text(
                        deterministic_first_proposal
                    )
                else:
                    round_payload = dict(prompt_payload)
                    if typed_feedback:
                        round_payload["typed_verifier_feedback"] = list(
                            typed_feedback
                        )
                    strong_message = generate(
                        model=strong_model,
                        messages=[
                            SystemMessage(
                                role="system",
                                content=CFRC_STRONG_SYSTEM_PROMPT,
                            ),
                            UserMessage.text(
                                json.dumps(
                                    round_payload,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                )
                            ),
                        ],
                        call_name=(
                            "cfrc_strong_residual_proposal"
                            if round_index == 1
                            else f"cfrc_strong_residual_proposal_round_{round_index}"
                        ),
                        **dict(strong_llm_args),
                    )
                    self._strong_calls_used += 1
                projection_start = time.perf_counter()
                proposal = parse_typed_proposal(strong_message.content or "")
                if (
                    freeze_completion is not None
                    and request_contract is not None
                    and not request_contract.obligations
                    and not proposal.parse_error
                    and proposal.nodes
                ):
                    frozen_pair = freeze_completion(proposal.nodes)
                    if frozen_pair is not None:
                        request_contract, certificate, contract_row = (
                            frozen_pair
                        )
                        visible_evidence = [
                            *[
                                row
                                for row in visible_evidence
                                if str(row.get("kind") or "")
                                != "request_scoped_obligation_contract"
                            ],
                            contract_row,
                        ]
                result = project_cfrc_proposal(
                    parameter_grounding_texts=_parameter_grounding_texts(self.tools),
                    certificate=certificate,
                    proposal=proposal,
                    available_tool_names=available_tool_names,
                    required_carrier_tool_calls=(
                        self._pending_frontier_tool_calls
                        if frontier_timing == "pre_write_candidate"
                        and request_contract is None
                        else []
                    ),
                    visible_evidence=visible_evidence,
                    compiler_setting=compiler_setting,
                    compiler_suffix_tool_call_budget=(
                        effective_suffix_tool_call_budget
                        if effective_suffix_tool_call_budget is not None
                        else suffix_tool_call_budget
                    ),
                )
                contract_coverage = None
                contract_literals_deterministically_carried = False
                deterministic_literal_gap_requires_evidence = False
                deterministic_terminal_text = ""
                replica_witnesses: dict[str, tuple[str, bool]] = {}
                # Round local. The replica runs at most once per round and not
                # at all when the compiler already blocked, so reading the
                # agent attribute directly would carry a previous round's
                # refusal into a round that never reached the replica.
                replica_failure_detail: dict[str, Any] = {}
                if request_contract is not None:
                    coverage_nodes = (
                        proposal.nodes
                        if request_semantics_builder
                        else result.selected_nodes
                        if request_contract_private_replica and result.execute
                        else proposal.nodes
                    )
                    proposal_hypothetical_calls = [
                        *self._contract_calls,
                        *[
                            ContractCall(
                                node_id=f"proposal:{node.node_id}",
                                tool_name=node.tool_name,
                                arguments=node.arguments,
                                succeeded=True,
                            )
                            for node in coverage_nodes
                            if node.kind == "tool_call"
                        ],
                    ]
                    proposal_terminal_text = "\n".join(
                        node.message
                        for node in proposal.nodes
                        if node.kind == "message"
                    )
                    contract_coverage = match_contract_calls(
                        contract=request_contract,
                        calls=proposal_hypothetical_calls,
                        write_tool_names=sorted(self._contract_allowed_tool_names),
                        terminal_text=proposal_terminal_text,
                    )
                    terminal_literal_gap = bool(
                        not contract_coverage.complete
                        and not contract_coverage.missing_obligation_ids
                        and not contract_coverage.extra_write_node_ids
                        and not contract_coverage.dependency_errors
                        and contract_coverage.missing_terminal_literals
                    )
                    literals_visibly_grounded = bool(
                        terminal_literal_gap
                        and _literals_visibly_grounded(
                            contract_coverage.missing_terminal_literals,
                            getattr(self._latest_state, "messages", []) or [],
                            derived_evidence=visible_evidence,
                            public_policy=str(
                                getattr(self, "domain_policy", "") or ""
                            ),
                        )
                    )
                    deterministic_literal_gap_requires_evidence = bool(
                        deterministic_round
                        and terminal_literal_gap
                        and not literals_visibly_grounded
                    )
                    if terminal_literal_gap and literals_visibly_grounded:
                        # Every write obligation is covered exactly and the
                        # only defect is contract literals the proposal's
                        # terminal fails to carry.  Abstaining here spends
                        # the whole verifier and proposal budget without a
                        # Checked Execution; the literal carriage is
                        # completed deterministically at the ledger-closed
                        # terminal instead.  Only literals already present
                        # in the visible conversation qualify: the parser
                        # does not ground required_terminal_literals, and a
                        # runner-authored sentence must not be the first
                        # place an ungrounded value appears.
                        amended_terminal_text = "\n".join(
                            row
                            for row in (
                                proposal_terminal_text,
                                _terminal_literal_sentence(
                                    contract_coverage.missing_terminal_literals
                                ),
                            )
                            if row
                        )
                        recomputed = match_contract_calls(
                            contract=request_contract,
                            calls=proposal_hypothetical_calls,
                            write_tool_names=sorted(self._contract_allowed_tool_names),
                            terminal_text=amended_terminal_text,
                        )
                        if recomputed.complete:
                            contract_coverage = recomputed
                            contract_literals_deterministically_carried = True
                            deterministic_terminal_text = amended_terminal_text
                    if not contract_coverage.complete:
                        result = _abstain(
                            certificate,
                            proposal,
                            reason="request_contract_incomplete_proposal",
                            projection={
                                "decision": "project_abstain",
                                "request_contract": request_contract.public_json(),
                                "request_contract_coverage": (
                                    contract_coverage.public_json()
                                ),
                            },
                        )
                    else:
                        if (
                            request_contract_private_replica
                            and result.execute
                            and (
                                contract_builder_variant
                                != "model_authored_request_semantics"
                                or any(
                                    node.kind == "tool_call"
                                    for node in result.selected_nodes
                                )
                            )
                        ):
                            replica_witnesses, replica_error = (
                                self._proposal_replica_witnesses(
                                    result.selected_nodes
                                )
                            )
                            if replica_error:
                                replica_failure_detail = dict(
                                    self._last_replica_failure
                                )
                            coverage_by_node: dict[str, list[str]] = {}
                            for (
                                obligation_id,
                                covered_node_id,
                            ) in contract_coverage.covered_by_obligation.items():
                                node_id = str(covered_node_id)
                                if not node_id.startswith("proposal:"):
                                    continue
                                node_id = node_id.removeprefix("proposal:")
                                coverage_by_node.setdefault(node_id, []).append(
                                    obligation_id
                                )
                            witnessed_result = project_cfrc_proposal(
                                parameter_grounding_texts=_parameter_grounding_texts(self.tools),
                                certificate=certificate,
                                proposal=proposal,
                                available_tool_names=available_tool_names,
                                required_carrier_tool_calls=[],
                                visible_evidence=visible_evidence,
                                compiler_setting=compiler_setting,
                                compiler_suffix_tool_call_budget=(
                                    effective_suffix_tool_call_budget
                                    if effective_suffix_tool_call_budget is not None
                                    else suffix_tool_call_budget
                                ),
                                replica_witnesses=replica_witnesses,
                                coverage_by_node=coverage_by_node,
                                require_action_receipt_witness=True,
                            )
                            result = CFRCProjectionResult(
                                execute=witnessed_result.execute,
                                selected_nodes=witnessed_result.selected_nodes,
                                certificate=witnessed_result.certificate,
                                proposal=witnessed_result.proposal,
                                projection={
                                    **dict(witnessed_result.projection),
                                    "private_reached_state_replica": not bool(
                                        replica_error
                                    ),
                                    "proposal_time_replica_receipt_count": len(
                                        replica_witnesses
                                    ),
                                    "live_replica_receipt_match_required": True,
                                    "private_replica_error_code": replica_error,
                                },
                                reason=(
                                    witnessed_result.reason
                                    if not replica_error
                                    else replica_error
                                ),
                            )
                        result = CFRCProjectionResult(
                            execute=result.execute,
                            selected_nodes=result.selected_nodes,
                            certificate=result.certificate,
                            proposal=result.proposal,
                            projection={
                                **dict(result.projection),
                                "request_contract": request_contract.public_json(),
                                "request_contract_coverage": (
                                    contract_coverage.public_json()
                                ),
                                "request_contract_literals_deterministically_carried": (
                                    contract_literals_deterministically_carried
                                ),
                            },
                            reason=result.reason,
                        )
                projection_seconds = time.perf_counter() - projection_start
                rounds.append(
                    {
                        "round_index": round_index,
                        "strong_message": strong_message,
                        "proposal": proposal,
                        "result": result,
                        "projection_seconds": projection_seconds,
                        "deterministic_transcription": deterministic_round,
                        "deterministic_terminal_text": deterministic_terminal_text,
                        "replica_witnesses": replica_witnesses,
                        "replica_failure": replica_failure_detail,
                    }
                )
                budget_spent = (
                    self._strong_calls_used >= call_cap
                    if request_contract is not None
                    else len(rounds) >= call_cap
                )
                transcription_pending = transcription_is_fallback and bool(
                    deterministic_first_proposal is not None
                    and not any(
                        row["deterministic_transcription"] for row in rounds
                    )
                )
                if (
                    result.execute
                    or (budget_spent and not transcription_pending)
                    or deterministic_literal_gap_requires_evidence
                ):
                    break
                if request_contract is not None and contract_coverage is not None:
                    typed_feedback.append(
                        contract_feedback(request_contract, contract_coverage)
                    )
                typed_feedback.append(
                    _generic_typed_verifier_feedback(
                        round_index=round_index,
                        proposal=proposal,
                        result=result,
                        replica_failure=replica_failure_detail,
                    )
                )
            return rounds

    def factory(tools: Any, domain_policy: str, llm: str, llm_args: Any = None, **kwargs: Any):
        return Tau2CFRCAgent(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args or {},
            task=kwargs.get("task"),
        )

    if registry.get_agent_factory(agent_name) is None:
        registry.register_agent_factory(factory, agent_name)
    return agent_name


def _select_compiler_constrained_round(
    rounds: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if not rounds:
        raise ValueError("strong_proposal_rounds_empty")
    for row in rounds:
        result = row.get("result")
        if isinstance(result, CFRCProjectionResult) and result.execute:
            return row
    return rounds[0]


def _assistant_message_has_payload(message: Any) -> bool:
    content = getattr(message, "content", None)
    if content is not None and str(content).strip():
        return True
    return bool(_message_tool_calls(message))


def _public_cfrc_metadata(message: Any) -> dict[str, Any]:
    raw_data = getattr(message, "raw_data", None)
    if not isinstance(raw_data, dict):
        return {}
    papc_cfrc = raw_data.get("papc_cfrc")
    return dict(papc_cfrc) if isinstance(papc_cfrc, dict) else {}


def _attach_cfrc_fallback_metadata(
    message: Any,
    result: CFRCProjectionResult,
    *,
    fallback_source: str,
) -> Any:
    metadata = result.public_json()
    metadata["fallback_to_cheap_control"] = True
    metadata["fallback_source"] = fallback_source
    return _attach_cfrc_metadata(message, metadata)


def _released_candidate_copy(message: Any) -> Any:
    """A fresh message for a withheld candidate the method has decided to release.

    The candidate is the cheap model's own message, held by the agent while the
    frontier decided. Releasing it returned that same object, and the metadata
    helper below writes into the object it is given, so the transcript could
    come to hold one message twice. The evaluator walks the transcript
    expecting the results of a tool call to follow it, finds the second copy of
    the call where the first result should be, and refuses to score the episode
    at all: a Retail row failed this way on two separate attempts and was
    recorded both times as an infrastructure error it never was.

    A copy cannot alias anything already recorded. Fresh call identifiers keep
    it from colliding with an earlier emission of the same call, which the
    evaluator checks separately. The tool and its arguments are untouched, so
    the action released is the action the candidate proposed.
    """

    if message is None:
        return None
    copier = getattr(message, "model_copy", None)
    released = copier(deep=True) if callable(copier) else copy.deepcopy(message)
    calls = getattr(released, "tool_calls", None)
    if not calls:
        # An empty list still reads as a tool call to the evaluator, which then
        # expects results for a call that was never made. Absent says what is
        # meant.
        try:
            released.tool_calls = None
        except Exception:
            pass
        return released
    for call in calls:
        try:
            call.id = f"call_{secrets.token_hex(12)}"
        except Exception:
            pass
    return released


def _attach_cfrc_metadata(
    message: Any,
    metadata: Mapping[str, Any],
) -> Any:
    metadata = dict(metadata)
    raw_data = getattr(message, "raw_data", None)
    if isinstance(raw_data, dict):
        raw_data["papc_cfrc"] = metadata
        return message
    try:
        setattr(message, "raw_data", {"papc_cfrc": metadata})
    except Exception:
        pass
    return message


def _graph_residual_continuation_policy(
    *,
    result: CFRCProjectionResult,
    has_tool_suffix: bool,
    fallback_after_tool_suffix_receipt: bool,
    fallback_after_abstain: bool,
) -> dict[str, Any]:
    after_residual_suffix = bool(
        result.execute and has_tool_suffix and fallback_after_tool_suffix_receipt
    )
    after_abstain = bool((not result.execute) and fallback_after_abstain)
    if after_residual_suffix:
        policy = "continue_cheap_after_residual_suffix_receipt"
    elif after_abstain:
        policy = "resume_cheap_after_graph_core_abstain"
    else:
        policy = "terminal_residual_output"
    return {
        "schema": "papc.tau2.graph_residual_continuation.v1",
        "enabled": after_residual_suffix or after_abstain,
        "after_residual_suffix": after_residual_suffix,
        "after_abstain": after_abstain,
        "policy": policy,
        "layer": "graph_residual_continuation",
        "strict_core_counted": False,
    }


def _generic_typed_verifier_feedback(
    *,
    round_index: int,
    proposal: CFRCProposalGraph,
    result: CFRCProjectionResult,
    replica_failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """What the successor is told before it is asked to propose again.

    Checked Execution is a three-valued verdict followed by bounded
    re-proposal, and a re-proposal that cannot see the verdict is resampling
    rather than repair. A suffix that the compiler admits and the private
    replica then refuses arrives here as ``projection_execute`` false beside a
    rejected-node count of zero, because no node was rejected: the graph was
    well formed and the action did not run against the reached state. Three
    rounds read that as no information and re-proposed the same class of
    suffix. The verdict below distinguishes the two failures and says which
    node stopped the suffix, using a position the successor itself authored
    and an exception class name. It does not name a tool, an argument, or a
    replacement value, so the successor learns that its action was refused
    without being told what to write instead.
    """

    diagnostics = _mapping_or_none(result.projection.get("diagnostics")) or {}
    failure = _mapping_or_none(replica_failure) or {}
    feedback = {
        "schema": "papc.tau2.generic_typed_verifier_feedback.v2",
        "round_index": int(round_index),
        "projection_execute": bool(result.execute),
        "candidate_node_count": len(proposal.nodes),
        "projected_node_count": len(result.selected_nodes),
        "rejected_node_count": int(diagnostics.get("frontier_rejected_node_count") or 0),
        "replica_execution_refused": bool(failure),
        "instruction": (
            "Re-propose a complete typed graph or abstain. Do not change z, "
            "do not assume hidden facts, and cite only visible evidence."
        ),
        "contains_tool_names": False,
        "contains_argument_values": False,
        "contains_failure_shape_hint": False,
    }
    if failure:
        feedback["failed_node_position"] = int(
            failure.get("failed_node_position") or 0
        )
        feedback["tool_node_count"] = int(failure.get("tool_node_count") or 0)
        feedback["error_type"] = str(failure.get("error_type") or "")
        feedback["instruction"] = (
            "The previous suffix was well formed and its action at the node "
            "position above did not execute against the reached state. Do not "
            "repeat that action unchanged. Either propose a suffix whose "
            "actions the reached state admits, cite the visible evidence that "
            "would make the same action admissible, or abstain. Do not change "
            "z and do not assume hidden facts."
        )
    return feedback


def _write_agent_decision_row(
    *,
    path: Path,
    task: Any,
    frontier_timing: FrontierTiming,
    cheap_sidecar_message: Any | None,
    strong_messages: Sequence[Any],
    result: CFRCProjectionResult,
    strong_rounds: Sequence[Mapping[str, Any]] = (),
    strong_call_cap_configured: int | None = None,
    contract_builder_variant: str = "deterministic_candidate_grounded",
    contract_builder_attempt_cap_configured: int = 0,
    contract_builder_phase_names: Sequence[str] = (),
    closure_rejection_reasons: Mapping[str, int] | None = None,
    contract_builder_ledger_role: Literal["strong", "cheap"] = "strong",
    total_strong_provider_call_cap_configured: int | None = None,
    frontier_candidate_private: Sequence[Mapping[str, Any]] = (),
    projection_seconds: float | None = None,
    residual_activation_reason: str | None = None,
    candidate_tool_call_executed_before_frontier: bool | None = None,
    receipt_summary: Mapping[str, Any] | None = None,
    graph_residual_continuation: Mapping[str, Any] | None = None,
) -> None:
    ledger = RuntimeLedger()
    if cheap_sidecar_message is not None:
        ledger.add_usage(
            "cheap",
            usage_with_raw_cache_details(cheap_sidecar_message),
            latency_seconds=_optional_float(
                getattr(cheap_sidecar_message, "generation_time_seconds", None)
            ),
            cost_usd=_optional_float(getattr(cheap_sidecar_message, "cost", None)),
        )
    # Contract-builder messages always lead ``strong_messages``; when the
    # builder is bound to the cheap prefix model its usage belongs to the cheap
    # role, so the strong ledger keeps measuring residual completion only.
    builder_message_count = (
        len(contract_builder_phase_names)
        if contract_builder_ledger_role == "cheap"
        else 0
    )
    for index, strong_message in enumerate(strong_messages):
        ledger.add_usage(
            contract_builder_ledger_role
            if index < builder_message_count
            else "strong",
            usage_with_raw_cache_details(strong_message),
            latency_seconds=_optional_float(
                getattr(strong_message, "generation_time_seconds", None)
            ),
            cost_usd=_optional_float(getattr(strong_message, "cost", None)),
        )
    loop_audit = _proposal_loop_audit(
        strong_rounds=strong_rounds,
        selected_result=result,
        strong_call_cap_configured=strong_call_cap_configured,
    )
    actual_residual_proposal_calls = sum(
        1
        for round_row in strong_rounds
        if not round_row.get("deterministic_transcription")
        and round_row.get("strong_message") is not None
    )
    builder_phase_names = [str(name) for name in contract_builder_phase_names]
    builder_phase_ledgers: dict[str, RuntimeLedger] = {}
    for phase, strong_message in zip(
        builder_phase_names,
        strong_messages,
        strict=False,
    ):
        phase_ledger = builder_phase_ledgers.setdefault(phase, RuntimeLedger())
        phase_ledger.add_usage(
            contract_builder_ledger_role,
            usage_with_raw_cache_details(strong_message),
            latency_seconds=_optional_float(
                getattr(strong_message, "generation_time_seconds", None)
            ),
            cost_usd=_optional_float(getattr(strong_message, "cost", None)),
        )
    actual_contract_builder_calls = (
        len(builder_phase_names)
        if builder_phase_names
        else max(
            0,
            len(strong_messages) - actual_residual_proposal_calls,
        )
    )
    residual_proposal_call_cap = loop_audit["B_call"]
    total_strong_provider_call_cap = int(
        total_strong_provider_call_cap_configured
        if total_strong_provider_call_cap_configured is not None
        else int(contract_builder_attempt_cap_configured)
        + int(residual_proposal_call_cap)
    )
    candidate_executed = (
        bool(candidate_tool_call_executed_before_frontier)
        if candidate_tool_call_executed_before_frontier is not None
        else frontier_timing == "post_tool_result"
    )
    residual_activation_reason = residual_activation_reason or (
        "post_write_residual_activation_always_after_receipt"
        if frontier_timing == "post_tool_result"
        else "pre_write_residual_activation_always"
    )
    row = {
        "schema": "papc.tau2.cfrc_agent_decision.v1",
        "task_hash": _stable_public_hash(str(task)),
        "protocol": "graph_cfrc_core_then_explicit_continuation_policy",
        "method_layer": "graph_cfrc_core",
        "frontier_verifier": {
            "schema": "papc.tau2.cfrc_frontier_verifier.v1",
            "accepted": True,
            "predicate": (
                "pre_write_candidate_first_mutating_tool_call"
                if frontier_timing == "pre_write_candidate"
                else "post_tool_result_first_mutating_tool_call"
            ),
            "frontier_timing": frontier_timing,
            "frontier_tool_names": list(result.certificate.frontier_tool_names),
            "committed_tool_count": max(
                len(result.certificate.committed_tools),
                len(result.certificate.commitments),
            ),
            "legacy_committed_tool_count": len(result.certificate.committed_tools),
            "receipt_bound_commitment_count": len(result.certificate.commitments),
            "evidence_event_count": result.certificate.evidence_event_count,
            "candidate_tool_call_executed_before_frontier": candidate_executed,
            "residual_activation_reason": residual_activation_reason,
            "carrier_exact_match_required": (
                frontier_timing == "pre_write_candidate"
                and contract_builder_variant
                == "deterministic_candidate_grounded"
            ),
            "extra_node_binding": (
                "same_tool_same_arg_schema_cited_visible_evidence_or_carrier"
                if frontier_timing == "pre_write_candidate"
                and contract_builder_variant
                == "deterministic_candidate_grounded"
                else (
                    "contract_only_no_carrier"
                    if contract_builder_variant
                    == "model_authored_request_semantics"
                    else "disabled"
                )
            ),
            "cheap_can_decide_escalation": False,
        },
        "frontier_candidate_private": list(frontier_candidate_private),
        "contract_builder": {
            "schema": "papc.tau2.contract_builder_budget.v1",
            "variant": contract_builder_variant,
            "activation_scope": "accepted_first_write_commitment_frontier",
            "attempt_cap": int(contract_builder_attempt_cap_configured),
            "successful_construction_provider_call_count_required": int(
                contract_builder_attempt_cap_configured
            ),
            "actual_provider_call_count": actual_contract_builder_calls,
            "phases": [
                {
                    "phase": phase,
                    "provider_call_count": builder_phase_names.count(phase),
                    "runtime_ledger": builder_phase_ledgers[
                        phase
                    ].public_summary(),
                }
                for phase in (
                    "request_semantics_closure",
                    "request_semantics_draft",
                    "residual_closure_revision",
                    "deterministic_residual_closure",
                    "deterministic_residual_closure_rejected",
                )
                if phase in builder_phase_names
            ],
            "final_contract_source": (
                "request_semantics_closure"
                if "request_semantics_closure" in builder_phase_names
                and "request_contract" in result.projection
                else "residual_closure_revision"
                if "residual_closure_revision" in builder_phase_names
                and "request_contract" in result.projection
                else "deterministic_grounding_plus_residual_closure"
                if "deterministic_residual_closure" in builder_phase_names
                else "deterministic_candidate_grounded"
                if contract_builder_variant
                == "deterministic_candidate_grounded"
                else "unavailable_fail_closed"
            ),
            # Counted, not assigned: the parse-error slot is overwritten by
            # whichever contract step runs last, so a closure the runtime paid
            # for and then discarded leaves no record of why unless it is
            # accumulated here.
            "closure_rejection_reasons": dict(
                sorted((closure_rejection_reasons or {}).items())
            ),
        },
        "strong_provider_call_budget": {
            "schema": "papc.tau2.strong_provider_call_budget.v1",
            "contract_builder_attempt_cap": int(
                contract_builder_attempt_cap_configured
            ),
            "contract_builder_phase_call_counts": {
                phase: builder_phase_names.count(phase)
                for phase in (
                    "request_semantics_closure",
                    "request_semantics_draft",
                    "residual_closure_revision",
                    "deterministic_residual_closure",
                    "deterministic_residual_closure_rejected",
                )
                if phase in builder_phase_names
            },
            "contract_builder_success_provider_call_count_required": int(
                contract_builder_attempt_cap_configured
            ),
            "residual_proposal_call_cap": int(residual_proposal_call_cap),
            "total_strong_provider_call_cap": total_strong_provider_call_cap,
            "actual_contract_builder_call_count": actual_contract_builder_calls,
            "actual_residual_proposal_call_count": (
                actual_residual_proposal_calls
            ),
            "actual_total_strong_provider_call_count": len(strong_messages),
        },
        "strong_call_count": len(strong_messages),
        "strong_call_cap": loop_audit["B_call"],
        "proposal_loop": loop_audit,
        "projection_result": result.public_json(),
        "graph_residual_continuation": dict(graph_residual_continuation or {}),
        "strong_proposal_private": result.proposal.private_json(),
        "strong_rounds_private": [
            {
                "round_index": int(round_row.get("round_index") or index + 1),
                "projection_result": round_row["result"].public_json(),
                "strong_proposal_private": round_row["proposal"].private_json(),
                "selected_for_execution": round_row["result"] is result,
                "deterministic_transcription": bool(
                    round_row.get("deterministic_transcription")
                ),
            }
            for index, round_row in enumerate(strong_rounds)
            if isinstance(round_row.get("result"), CFRCProjectionResult)
            and isinstance(round_row.get("proposal"), CFRCProposalGraph)
        ],
        "runtime_ledger": ledger.public_summary(),
        "deterministic_projection_seconds": (
            round(projection_seconds, 6) if projection_seconds is not None else None
        ),
        "residual_suffix_tool_execution_seconds": None,
        "residual_suffix_tool_execution_latency_available": False,
        "raw_messages_excluded": True,
        "public_outputs_exclude_private_diagnostics": True,
        "public_safe": False,
    }
    if receipt_summary is not None:
        row["receipt_summary"] = dict(receipt_summary)
    write_jsonl_append(path, row)


def _proposal_loop_audit(
    *,
    strong_rounds: Sequence[Mapping[str, Any]],
    selected_result: CFRCProjectionResult,
    strong_call_cap_configured: int | None,
) -> dict[str, Any]:
    configured_cap = max(1, int(strong_call_cap_configured or len(strong_rounds) or 1))
    actual_rounds = [
        round_row
        for round_row in strong_rounds
        if isinstance(round_row.get("result"), CFRCProjectionResult)
    ]
    first_result = actual_rounds[0]["result"] if actual_rounds else None
    selected_round_index = None
    for round_row in actual_rounds:
        if round_row["result"] is selected_result:
            selected_round_index = int(round_row.get("round_index") or 0)
            break
    return {
        "enabled": configured_cap > 1,
        "B_call": configured_cap,
        "actual_strong_call_count": len(actual_rounds),
        "selection_policy": "first_compiler_executable_round_else_one_shot_abstain",
        # Declared separately from the three flags below, which say the
        # feedback never carries a tool name, an argument value, or the shape
        # of a correct answer, and still do not. This says the feedback now
        # carries the Checked Execution verdict itself: whether the private
        # replica refused the suffix, and at which node position it stopped.
        # Without it a refused round is indistinguishable from a round the
        # compiler rejected on structure, and bounded re-proposal degenerates
        # into resampling the same suffix.
        "feedback_contains_execution_verdict": True,
        "replica_refused_round_count": sum(
            1
            for round_row in actual_rounds
            if _mapping_or_none(round_row.get("replica_failure"))
        ),
        # A refused round is currently only counted, so a run reports how often
        # the replica stopped a suffix but never why. The exception class and
        # the position it stopped at are both structural: neither names a tool,
        # an argument, or a value, so they carry no answer content and can be
        # published. Without them the only way to tell a malformed argument
        # apart from a state precondition the successor could not see is to run
        # the arm again with a debugger attached.
        "replica_failure_signatures": [
            {
                "error_type": str(detail.get("error_type") or ""),
                "failed_node_position": int(detail.get("failed_node_position") or 0),
                "tool_node_count": int(detail.get("tool_node_count") or 0),
            }
            for round_row in actual_rounds
            for detail in [_mapping_or_none(round_row.get("replica_failure"))]
            if detail
        ],
        "best_of_n_selection": False,
        "every_round_compiler_projected": len(actual_rounds) == len(strong_rounds),
        "one_shot_comparator_present": first_result is not None,
        "one_shot_execute": bool(first_result.execute) if first_result is not None else False,
        "one_shot_selected_node_count": (
            len(first_result.selected_nodes) if first_result is not None else 0
        ),
        "selected_round_index": selected_round_index,
        "stopped_after_first_executable": (
            first_result is selected_result
            and bool(first_result.execute)
            and len(actual_rounds) == 1
            and configured_cap > 1
        )
        if first_result is not None
        else False,
        "feedback": "generic_typed_counts_only" if configured_cap > 1 else "disabled",
        "feedback_contains_tool_names": False,
        "feedback_contains_argument_values": False,
        "feedback_contains_failure_shape_hint": False,
    }


def _runtime_role_for_tau_message(
    message: Mapping[str, Any],
    *,
    cheap_model: str,
    strong_model: str,
) -> str | None:
    role = str(message.get("role") or "").lower()
    if role == "user":
        return "user_simulator"
    if role != "assistant":
        return None
    raw = _mapping_or_none(message.get("raw_data")) or {}
    papc_cfrc = _mapping_or_none(raw.get("papc_cfrc"))
    if papc_cfrc:
        return None
    model = str(raw.get("model") or message.get("model") or "").lower()
    if cheap_model and cheap_model in model:
        return "cheap"
    if strong_model and strong_model in model:
        return "strong"
    return None


def _message_tool_calls(message: Any) -> list[Any]:
    if isinstance(message, Mapping):
        raw_tool_calls = message.get("tool_calls")
        if not isinstance(raw_tool_calls, list):
            return []
        calls: list[Any] = []
        for raw_call in raw_tool_calls:
            call = _tool_call_view_from_mapping(raw_call)
            if call is not None:
                calls.append(call)
        return calls
    tool_calls = getattr(message, "tool_calls", None)
    return list(tool_calls) if isinstance(tool_calls, list) else []


def _pre_write_candidate_requires_residual_activation(
    frontier: Sequence[Any],
    messages: Sequence[Any],
    *,
    policy: PreWriteResidualPolicy,
) -> bool:
    return bool(
        _pre_write_candidate_residual_activation_reason(
            frontier,
            messages,
            policy=policy,
        )
    )


def _pre_write_candidate_residual_activation_reason(
    frontier: Sequence[Any],
    messages: Sequence[Any],
    *,
    policy: PreWriteResidualPolicy,
) -> str | None:
    # The commitment frontier is the moment the cheap agent is about to
    # commit its first state effect, which is exactly what the formal
    # setting defines; no tool shape, item count, or read count enters the
    # decision.  Earlier revisions filtered candidates by hand-tuned retail
    # shapes, which fitted particular rows rather than the setting.
    del messages
    return (
        "pre_write_candidate_commitment_frontier" if frontier else None
    )




def _visible_assistant_tool_call_counts(messages: Sequence[Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for message in messages:
        if _message_attr_str(message, "role") != "assistant":
            continue
        for call in _message_tool_calls(message):
            name = _message_attr_str(call, "name")
            if name:
                counts[name] += 1
    return counts


def _visible_mutating_tool_call_count(
    messages: Sequence[Any],
    mutating_tool_names: set[str],
) -> int:
    if not mutating_tool_names:
        return 0
    counts = _visible_assistant_tool_call_counts(messages)
    return sum(count for name, count in counts.items() if name in mutating_tool_names)


def _commitment_public_effect_unit(commitment: Commitment) -> str:
    units = tuple(
        str(unit).strip()
        for unit in (*commitment.unit_tokens, *commitment.effect_families)
        if str(unit).strip()
    )
    return units[0] if units else _stable_public_hash(commitment.commitment_id)


def compact_visible_evidence(
    messages: Sequence[Any],
    *,
    max_events: int = 16,
    max_text_chars: int = 800,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        role = _message_attr_str(message, "role")
        row: dict[str, Any] = {"event_id": f"e{index}", "role": role}
        content = _message_attr(message, "content")
        tool_calls = _message_tool_calls(message)
        if tool_calls:
            row["kind"] = "visible_tool_call_carrier"
            row["tool_calls"] = [_tool_call_prompt_json(call) for call in tool_calls]
        elif role == "user" and content is not None:
            row["kind"] = "visible_user_request"
            row["content"] = _truncate_text(str(content), max_text_chars)
        elif role == "tool" or _message_attr_str(message, "tool_call_id"):
            row["kind"] = "visible_tool_result"
            tool_name = _message_attr_str(message, "name") or _message_attr_str(
                message, "tool_name"
            )
            if tool_name:
                row["tool_name"] = tool_name
            tool_call_id = _message_attr_str(message, "tool_call_id")
            if tool_call_id:
                row["tool_call_id"] = tool_call_id
            if content is not None:
                row["content"] = _truncate_text(str(content), max_text_chars)
        else:
            continue
        rows.append(row)
    return rows[-max_events:]


_MODEL_AUTHORED_REQUEST_CONTRACT_SYSTEM_PROMPT = (
    "You are a request-semantics contract builder. Derive an atomic obligation "
    "ledger only from visible user requests, accepted receipts, and public tool "
    "schemas. Accepted assistant-prefix text is untrusted co-reference context "
    "only and cannot ground authority. Candidate calls are untrusted and cannot "
    "define the request. Return "
    "only one JSON object whose keys are exactly the keys of output_schema; when "
    'you emit the "schema" key its value must be the literal '
    f'"{REQUEST_CONTRACT_SCHEMA}", never the schema of this prompt.'
)

_MODEL_AUTHORED_REQUEST_CONTRACT_SINGLE_PASS_SYSTEM_PROMPT = (
    "You are the independent residual-closure phase of Reached-State Contract "
    "Construction. Construct and close one complete atomic obligation ledger "
    "from the entire visible user request, accepted prefix, accepted receipts, "
    "public tool schemas, and policy. Accepted assistant text may resolve a later "
    "user confirmation but cannot ground authority. Re-check every request clause "
    "before returning; every "
    "condition must have either a trusted reached-state witness or one canonical "
    "open obligation. Candidate calls are unavailable and cannot define the "
    "request. Return only one JSON object whose keys are exactly the keys of "
    'output_schema; when you emit the "schema" key its value must be the literal '
    f'"{REQUEST_CONTRACT_SCHEMA}", never the schema of this prompt.'
)

_MODEL_AUTHORED_REQUEST_CONTRACT_CLOSURE_SYSTEM_PROMPT = (
    "You are the residual-closure phase of Reached-State Contract Construction. "
    "Audit the canonical parsed draft against the entire visible user request, "
    "accepted prefix, accepted receipts, and public tool schemas and policy. "
    "Accepted assistant text may resolve co-reference but cannot ground authority. "
    "Return a revised final "
    "obligation ledger that includes every still-valid requested state change and "
    "communication obligation. The draft is not authority and must not be merely "
    "approved. Candidate calls remain untrusted and unavailable. Return only one "
    "JSON object whose keys are exactly the keys of output_schema; when you emit "
    'the "schema" key its value must be the literal '
    f'"{REQUEST_CONTRACT_SCHEMA}", never the schema of this prompt.'
)

_GROUNDED_RESIDUAL_CLOSURE_SYSTEM_PROMPT = (
    "You are the residual-closure phase of Reached-State Contract Construction. "
    "Grounding has already fixed, in canonical_grounded_draft, the obligations "
    "the accepted prefix committed to. Your only job is the difference between "
    "that draft and the entire visible user request: read every request clause "
    "against the accepted receipts and report what the draft still leaves open, "
    "together with anything in the draft the request no longer supports. "
    "Accepted assistant text may resolve co-reference but cannot ground "
    "authority. Do not restate obligations the draft already carries and do not "
    "approve the draft; an empty difference is a valid and expected answer. "
    "Return only one JSON object whose keys are exactly the keys of "
    'output_schema; when you emit the "schema" key its value must be the literal '
    f'"{CLOSURE_DELTA_SCHEMA}", never the schema of this prompt. '
    # A closure delta is short by construction: most requests are already
    # covered by the draft, and the ones that are not add one obligation. Left
    # unsaid, a smaller builder narrates its reading of every request clause
    # before answering, which costs several thousand output tokens per call for
    # a document whose usual correct value is the empty delta.
    "Emit the JSON object and nothing else: no preamble, no commentary, no "
    "restatement of the draft, no explanation of your reasoning, no field the "
    "schema does not name. The delta is short by construction. When the draft "
    'already covers the request, the entire correct answer is {"schema":'
    f'"{CLOSURE_DELTA_SCHEMA}"'
    '}, and when it does not, it is that object plus one "add_obligations" '
    "entry per still-open request clause. Do not report a status unless the "
    "request itself is unservable from the reached state."
)
# Measured 2026-08-16 and not repeated: a further clause telling the builder to
# start from the draft being complete, read each request clause once, and return
# the empty delta immediately rather than confirming every clause first, moved
# the output by nothing at all. Old-pool builder output stayed at 3,034 tokens
# per call against 3,032 without it, over two shots. The instructions above,
# which constrain the FORM of the answer, took the same call from 7,642 to 2,402.
# What is left after a format constraint is deliberation, and deliberation does
# not follow prose. STATE reports the same split directly: 72 percent of its
# repair output is reasoning tokens, leaving 188 of prose to compress.


def _all_visible_user_text(messages: Sequence[Any], max_chars: int = 6000) -> str:
    rows = [
        str(_message_attr(message, "content") or "").strip()
        for message in messages
        if _message_attr_str(message, "role") == "user"
        and str(_message_attr(message, "content") or "").strip()
    ]
    return _truncate_text("\n\n".join(rows), max_chars)


def _untrusted_accepted_prefix_context(
    messages: Sequence[Any],
    *,
    max_events: int = 16,
    max_text_chars: int = 1200,
) -> list[dict[str, Any]]:
    """Keep accepted assistant text for co-reference, never value grounding."""

    rows: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if _message_attr_str(message, "role") != "assistant":
            continue
        if _message_tool_calls(message):
            continue
        content = str(_message_attr(message, "content") or "").strip()
        if not content:
            continue
        rows.append(
            {
                "event_id": f"accepted_prefix_{index}",
                "kind": "untrusted_confirmation_context",
                "authority": False,
                "content": _truncate_text(content, max_text_chars),
            }
        )
    return rows[-max_events:]


def _identifier_first_introduced_by_user(
    messages: Sequence[Any],
    identifier: str,
) -> bool:
    """Distinguish user-pinned ids from ids echoed after system discovery."""

    needle = str(identifier or "").strip().casefold()
    if not needle:
        return False
    for message in messages:
        content = str(_message_attr(message, "content") or "")
        tool_calls = _message_tool_calls(message)
        carrier = json.dumps(
            [_tool_call_prompt_json(call) for call in tool_calls],
            ensure_ascii=False,
            sort_keys=True,
        )
        if needle not in f"{content}\n{carrier}".casefold():
            continue
        return _message_attr_str(message, "role") == "user"
    return False


def _public_tau_tool_schemas(
    tools: Sequence[Any], *, project_catalog: bool = True
) -> list[dict[str, Any]]:
    """The callable surface as the contract builder reads it.

    The builder is given the callable contract -- names, types, enums, defaults,
    required flags -- and not the argument prose, which restates a policy it
    already holds in full. Sixteen retail tools are 2940 tokens whole and 1158
    projected, and the catalog rides on every builder call.
    """

    rows: list[dict[str, Any]] = []
    for tool in tools:
        schema = getattr(tool, "openai_schema", None)
        if isinstance(schema, Mapping):
            row = json.loads(json.dumps(dict(schema), default=str))
            if project_catalog:
                row = compact_tool_schema(row, keep_tool_description=False)
            rows.append(row)
        else:
            rows.append({"name": str(getattr(tool, "name", "") or "")})
    return rows


def _first_visible_user_text(messages: Sequence[Any], max_chars: int = 1200) -> str:
    for message in messages:
        if _message_attr_str(message, "role") != "user":
            continue
        content = _message_attr(message, "content")
        if content is None:
            continue
        return _truncate_text(str(content), max_chars)
    return ""


def _truncate_text(value: str, max_chars: int) -> str:
    if max_chars < 1:
        return ""
    if len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return value[:max_chars]
    return value[: max_chars - 3] + "..."


def _message_attr_str(message: Any, name: str) -> str:
    value = _message_attr(message, name)
    return str(value).strip() if value is not None else ""


def _message_attr(message: Any, name: str) -> Any:
    if isinstance(message, Mapping):
        return message.get(name)
    return getattr(message, name, None)


def _tool_call_view_from_mapping(raw_call: Any) -> _ToolCallView | None:
    if not isinstance(raw_call, Mapping):
        return None
    function = raw_call.get("function")
    function_payload = function if isinstance(function, Mapping) else {}
    name = str(raw_call.get("name") or function_payload.get("name") or "").strip()
    if not name:
        return None
    raw_arguments = raw_call.get("arguments")
    if raw_arguments is None:
        raw_arguments = function_payload.get("arguments")
    return _ToolCallView(
        id=str(raw_call.get("id") or raw_call.get("tool_call_id") or "").strip(),
        name=name,
        arguments=_tool_call_arguments_mapping(raw_arguments),
    )


def _tool_call_arguments_mapping(raw_arguments: Any) -> Mapping[str, Any]:
    if isinstance(raw_arguments, Mapping):
        return dict(raw_arguments)
    if isinstance(raw_arguments, str) and raw_arguments.strip():
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return {}


def _tool_call_prompt_json(call: Any) -> dict[str, Any]:
    row = {
        "name": _message_attr_str(call, "name"),
        "arguments": dict(_message_attr(call, "arguments") or {}),
    }
    call_id = _message_attr_str(call, "id")
    if call_id:
        row["tool_call_id"] = call_id
    return row


def _message_node_content(
    raw_node: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> str:
    message = raw_node.get("message")
    if isinstance(message, Mapping):
        return str(message.get("content") or message.get("text") or "").strip()
    if message is not None:
        return str(message).strip()
    content = raw_node.get("content")
    if content is not None:
        return str(content).strip()
    argument_message = arguments.get("message")
    if argument_message is not None:
        return str(argument_message).strip()
    return ""


def _compact_tool_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Use the shared projection for the successor's callable surface."""

    return compact_tool_schema(schema)


def _selected_tool_schemas(
    tools: Sequence[Any], tool_names: set[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tool in tools:
        name = str(getattr(tool, "name", ""))
        if name not in tool_names:
            continue
        schema = getattr(tool, "openai_schema", None)
        rows.append(
            _compact_tool_schema(schema) if isinstance(schema, dict) else {"name": name}
        )
    return rows


def _residual_tool_schemas(
    tools: Sequence[Any], mutating_tool_names: set[str]
) -> list[dict[str, Any]]:
    return _selected_tool_schemas(tools, mutating_tool_names)


def _support_read_tool_schemas(
    tools: Sequence[Any], support_read_tool_names: set[str]
) -> list[dict[str, Any]]:
    return _selected_tool_schemas(tools, support_read_tool_names)


def _schema_derived_ell3_restoration_metadata(
    *,
    tools: Sequence[Any],
    commitments: Sequence[Commitment],
    compiler_setting: CompilerSetting,
) -> dict[str, Any]:
    if _compiler_setting(compiler_setting) != AUTHORITY_LIFTED_ELL3_COMPILER_SETTING:
        return {"collision_budget": 0.0, "restoration_cost": {}}
    signals = _schema_reversibility_signals_by_family(tools)
    reversible_units = [
        unit
        for commitment in commitments
        if any(
            classify_reversibility(signals.get(family, _unknown_write_signal(family)))
            == "reversible"
            for family in commitment.effect_families
        )
        for unit in commitment.unit_tokens
    ]
    restoration_cost = restoration_cost_vector(
        reversible_units,
        signals,
        family_of=lambda _unit: _family_for_committed_unit(
            _unit,
            commitments=commitments,
        ),
    )
    restoration_cost = {
        unit: cost
        for unit, cost in restoration_cost.items()
        if 0.0 < cost < IRREVERSIBLE_DAMAGE_THRESHOLD
    }
    collision_budget = sum(restoration_cost.values()) if restoration_cost else 0.0
    return {
        "collision_budget": collision_budget,
        "restoration_cost": restoration_cost,
    }


def _schema_reversibility_signals_by_family(
    tools: Sequence[Any],
) -> dict[str, WriteUnitSignal]:
    signals: dict[str, WriteUnitSignal] = {}
    for tool in tools:
        name = str(getattr(tool, "name", "")).strip()
        if not name:
            continue
        schema = getattr(tool, "openai_schema", None)
        schema_text = _schema_semantic_text(schema if isinstance(schema, Mapping) else {})
        has_schema_signal = bool(schema_text.strip())
        low_name = name.lower()
        is_update = low_name.startswith(("modify", "update", "change", "exchange"))
        is_create_or_cancel = low_name.startswith(
            ("book", "cancel", "create", "remove", "return", "submit")
        )
        signals[tool_effect_unit(name)] = WriteUnitSignal(
            unit_family=tool_effect_unit(name),
            in_place_update_tool=is_update and has_schema_signal,
            money_or_points=looks_like_money_or_points(schema_text),
            fee_or_penalty=looks_like_fee_or_penalty(schema_text),
            lifecycle_confirm=looks_like_confirm_lifecycle(schema_text),
            create_or_cancel_only=(
                is_create_or_cancel and not is_update and has_schema_signal
            ),
            reason="tau2_tool_schema" if has_schema_signal else "schema_signal_missing",
        )
    return signals


def _schema_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        return " ".join(
            part
            for key, item in value.items()
            for part in (_schema_text(key), _schema_text(item))
            if part
        )
    if isinstance(value, (list, tuple)):
        return " ".join(part for item in value if (part := _schema_text(item)))
    return str(value)


def _schema_semantic_text(
    value: Any,
    *,
    _depth: int = 0,
    _parent_key: str = "",
) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        return " ".join(
            part
            for key, item in value.items()
            for part in (
                (
                    str(key).strip()
                    if _parent_key == "properties"
                    else (
                        str(key).strip()
                        if str(key).strip().lower() not in _SCHEMA_STRUCTURAL_KEYS
                        else ""
                    )
                ),
                (
                    ""
                    if _depth == 0 and str(key).strip().lower() == "name"
                    else _schema_semantic_text(
                        item,
                        _depth=_depth + 1,
                        _parent_key=str(key).strip().lower(),
                    )
                ),
            )
            if part
        )
    if isinstance(value, (list, tuple)):
        return " ".join(
            part
            for item in value
            if (
                part := _schema_semantic_text(
                    item,
                    _depth=_depth + 1,
                    _parent_key=_parent_key,
                )
            )
        )
    return str(value)


def _family_for_committed_unit(
    unit: str,
    *,
    commitments: Sequence[Commitment],
) -> str:
    for commitment in commitments:
        if unit in commitment.unit_tokens and commitment.effect_families:
            return commitment.effect_families[0]
    return unit


def _unknown_write_signal(unit_family: str) -> WriteUnitSignal:
    return WriteUnitSignal(unit_family=unit_family, reason="schema_signal_missing")


def _mutating_tool_names_from_tau_tools(
    tools: Sequence[Any],
    fallback_tool_names: Sequence[str],
) -> tuple[str, ...]:
    names: list[str] = []
    for tool in tools:
        name = str(getattr(tool, "name", ""))
        func = getattr(tool, "_func", None)
        mutates = getattr(func, "__mutates_state__", None)
        if mutates is True:
            names.append(name)
    if names:
        return normalize_tool_names(names)
    return mutating_tool_names_by_heuristic(fallback_tool_names)


def _projection_diagnostics(
    *,
    certificate: CFRCResidualCertificate,
    proposal: CFRCProposalGraph,
    available_tool_names: set[str],
    allowed_tool_names: set[str],
    frontier_binding: Mapping[str, Any] | None = None,
    visible_evidence: Sequence[Mapping[str, Any]] = (),
    carrier_tool_calls: Sequence[Mapping[str, Any]] = (),
    parameter_grounding_texts: Mapping[str, Mapping[str, str]] = MappingProxyType({}),
) -> dict[str, Any]:
    reject_counts: dict[str, int] = {}
    tool_call_node_count = 0
    read_call_node_count = 0
    message_node_count = 0
    structurally_valid_node_count = 0
    committed_effect_node_count = 0
    for node in proposal.nodes:
        if node.kind == "tool_call":
            tool_call_node_count += 1
        if node.kind == "read_call":
            read_call_node_count += 1
        if node.kind == "message":
            message_node_count += 1
        reasons = _typed_node_reject_reasons(
            node,
            available_tool_names=available_tool_names,
            allowed_tool_names=allowed_tool_names,
            message_residual_allowed=certificate.message_residual_allowed,
            visible_evidence=visible_evidence,
            carrier_tool_calls=carrier_tool_calls,
            parameter_grounding_texts=parameter_grounding_texts,
        )
        if reasons:
            for reason in reasons:
                reject_counts[reason] = reject_counts.get(reason, 0) + 1
        else:
            structurally_valid_node_count += 1
        if any(
            unit in certificate.contract_forbidden_units
            for unit in _node_write_units(
                node,
                certificate=certificate,
                carrier_tool_calls=carrier_tool_calls,
            )
        ):
            committed_effect_node_count += 1
    binding = dict(frontier_binding or {})
    if binding.get("carrier_exact_match_required") and binding.get("frontier_rejected_node_count"):
        reject_counts["not_matching_frontier_carrier"] = reject_counts.get(
            "not_matching_frontier_carrier", 0
        ) + int(binding.get("frontier_rejected_node_count") or 0)
    valid_typed_node_count = (
        int(binding.get("frontier_bound_node_count") or 0)
        if binding.get("carrier_exact_match_required")
        else structurally_valid_node_count
    )
    extra_fields = {}
    if binding.get("carrier_exact_match_required"):
        extra_fields = {
            "frontier_exact_match_node_count": int(
                binding.get("frontier_exact_match_node_count") or 0
            ),
            "frontier_extra_evidence_bound_node_count": int(
                binding.get("frontier_extra_evidence_bound_node_count") or 0
            ),
            "frontier_rejected_node_count": int(
                binding.get("frontier_rejected_node_count") or 0
            ),
        }
    return {
        "schema": "papc.tau2.cfrc_projection_diagnostics.v1",
        "abstain_bucket": _abstain_bucket(
            proposal=proposal,
            valid_typed_node_count=valid_typed_node_count,
            reject_counts=reject_counts,
        ),
        "candidate_node_count": len(proposal.nodes),
        "tool_call_node_count": tool_call_node_count,
        "read_call_node_count": read_call_node_count,
        "message_node_count": message_node_count,
        "valid_typed_node_count_before_pi": valid_typed_node_count,
        "committed_effect_node_count": committed_effect_node_count,
        "evidence_citation_reject_count": sum(
            count
            for reason, count in reject_counts.items()
            if reason in {
                "required_evidence_not_visible",
                "required_evidence_not_verified_carrier",
                "argument_value_not_in_cited_evidence",
            }
        ),
        "reject_reason_counts": dict(sorted(reject_counts.items())),
        **extra_fields,
    }


def _typed_node_reject_reasons(
    node: CFRCProposalNode,
    *,
    available_tool_names: set[str],
    allowed_tool_names: set[str],
    message_residual_allowed: bool = False,
    visible_evidence: Sequence[Mapping[str, Any]] = (),
    carrier_tool_calls: Sequence[Mapping[str, Any]] = (),
    parameter_grounding_texts: Mapping[str, Mapping[str, str]] = MappingProxyType({}),
) -> tuple[str, ...]:
    reasons: list[str] = []
    if node.kind == "message":
        if not message_residual_allowed:
            reasons.append("message_residual_not_allowed")
        if not node.message.strip():
            reasons.append("missing_message_content")
        if not node.claimed_effects:
            reasons.append("missing_claimed_effects")
        if not node.required_evidence:
            reasons.append("missing_required_evidence")
        reasons.extend(
            _evidence_citation_reject_reasons(
                node,
                visible_evidence=visible_evidence,
                carrier_tool_calls=carrier_tool_calls,
                parameter_grounding_texts=parameter_grounding_texts,
            )
        )
        return tuple(reasons)
    if node.kind not in {"tool_call", "read_call"}:
        reasons.append("non_tool_call_node")
        return tuple(reasons)
    if not node.tool_name:
        reasons.append("missing_tool_name")
    else:
        if node.tool_name not in available_tool_names:
            reasons.append("tool_not_available")
        if node.kind == "tool_call" and node.tool_name not in allowed_tool_names:
            reasons.append("tool_not_in_residual_schema")
    if not node.claimed_effects:
        reasons.append("missing_claimed_effects")
    if not node.required_evidence:
        reasons.append("missing_required_evidence")
    reasons.extend(
        _evidence_citation_reject_reasons(
            node,
            visible_evidence=visible_evidence,
            carrier_tool_calls=carrier_tool_calls,
            parameter_grounding_texts=parameter_grounding_texts,
        )
    )
    return tuple(reasons)


def _filter_nodes_by_frontier_binding(
    nodes: Sequence[CFRCProposalNode],
    *,
    carrier_counts: Counter[str],
    carrier_tool_calls: Sequence[Mapping[str, Any]],
    visible_evidence: Sequence[Mapping[str, Any]],
) -> tuple[tuple[CFRCProposalNode, ...], dict[str, Any]]:
    binding = {
        "carrier_exact_match_required": bool(carrier_counts),
        "frontier_bound_node_count": 0,
        "frontier_exact_match_node_count": 0,
        "frontier_extra_evidence_bound_node_count": 0,
        "frontier_rejected_node_count": 0,
    }
    if not carrier_counts:
        binding["frontier_bound_node_count"] = len(nodes)
        return tuple(nodes), binding
    remaining = Counter(carrier_counts)
    selected: list[CFRCProposalNode] = []
    for node in nodes:
        if node.kind != "tool_call":
            selected.append(node)
            continue
        signature = _tool_call_signature(node.tool_name, node.arguments)
        if remaining[signature] <= 0:
            if _extra_node_arguments_evidence_bound(
                node,
                carrier_tool_calls=carrier_tool_calls,
                visible_evidence=visible_evidence,
            ):
                selected.append(node)
                binding["frontier_extra_evidence_bound_node_count"] += 1
            else:
                binding["frontier_rejected_node_count"] += 1
            continue
        selected.append(node)
        remaining[signature] -= 1
        binding["frontier_exact_match_node_count"] += 1
    binding["frontier_bound_node_count"] = len(selected)
    return tuple(selected), binding


def _extra_node_arguments_evidence_bound(
    node: CFRCProposalNode,
    *,
    carrier_tool_calls: Sequence[Mapping[str, Any]],
    visible_evidence: Sequence[Mapping[str, Any]],
) -> bool:
    if node.kind != "tool_call":
        return False
    carrier_for_tool = [
        call
        for call in carrier_tool_calls
        if str(call.get("name") or "").strip() == node.tool_name
        and isinstance(call.get("arguments"), Mapping)
    ]
    if not carrier_for_tool:
        return False
    node_keys = set(str(key) for key in node.arguments)
    if not any(
        node_keys == set(str(key) for key in (_mapping_or_none(call.get("arguments")) or {}))
        for call in carrier_for_tool
    ):
        return False

    cited_rows = _cited_visible_evidence_rows(node.required_evidence, visible_evidence)
    if not cited_rows:
        return False
    carrier_values = {
        _normalize_argument_leaf(value)
        for call in carrier_for_tool
        for value in _argument_leaf_values(_mapping_or_none(call.get("arguments")) or {})
    }
    cited_values = _visible_evidence_leaf_value_set(cited_rows)
    uses_non_carrier_value = False
    for raw_value in _argument_leaf_values(node.arguments):
        value = _normalize_argument_leaf(raw_value)
        if not value:
            continue
        if value in carrier_values:
            if (
                CHEAP_FRONTIER_CANDIDATE_EVIDENCE
                not in _canonical_evidence_id_set(node.required_evidence)
            ):
                return False
            continue
        uses_non_carrier_value = True
        if value not in cited_values:
            return False
    return uses_non_carrier_value


def _argument_leaf_items(arguments: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """Yield (top-level argument name, leaf value) pairs."""

    items: list[tuple[str, Any]] = []
    for key, value in (arguments or {}).items():
        for leaf in _argument_leaf_values({key: value}):
            items.append((str(key), leaf))
    return items


def _parameter_grounding_texts(tools: Sequence[Any]) -> dict[str, dict[str, str]]:
    """Read each tool's published parameter text from the surface's own schema.

    The schema is identical for every task, so a value it names carries no
    task-specific information. Nothing here is surface-specific: whatever the
    environment publishes is what a closed-vocabulary argument may cite.
    """

    texts: dict[str, dict[str, str]] = {}
    for tool in tools:
        schema = getattr(tool, "openai_schema", None)
        function = (_mapping_or_none(schema) or {}).get("function")
        function = _mapping_or_none(function) or {}
        name = str(function.get("name") or getattr(tool, "name", "") or "")
        if not name:
            continue
        properties = (
            _mapping_or_none((_mapping_or_none(function.get("parameters")) or {}).get(
                "properties"
            ))
            or {}
        )
        per_parameter: dict[str, str] = {}
        for parameter, spec in properties.items():
            spec_map = _mapping_or_none(spec) or {}
            parts = [str(spec_map.get("description") or "")]
            enum_values = spec_map.get("enum")
            if isinstance(enum_values, list):
                parts.extend(str(item) for item in enum_values)
            text = " ".join(part for part in parts if part).lower()
            if text:
                per_parameter[str(parameter)] = text
        if per_parameter:
            texts[name] = per_parameter
    return texts


def _evidence_citation_reject_reasons(
    node: CFRCProposalNode,
    *,
    visible_evidence: Sequence[Mapping[str, Any]],
    carrier_tool_calls: Sequence[Mapping[str, Any]],
    parameter_grounding_texts: Mapping[str, Mapping[str, str]] = MappingProxyType({}),
) -> tuple[str, ...]:
    if not node.required_evidence:
        return ()
    visible_ids = {
        str(row.get("event_id") or "").strip()
        for row in visible_evidence
        if str(row.get("event_id") or "").strip()
    }
    fatal_reasons: list[str] = []
    missing_visible_ref = False
    cited_rows: list[Mapping[str, Any]] = []
    allow_carrier = False
    for raw_evidence_id in node.required_evidence:
        evidence_id = _canonical_evidence_id(raw_evidence_id)
        if evidence_id == CHEAP_FRONTIER_CANDIDATE_EVIDENCE:
            if carrier_tool_calls:
                allow_carrier = True
                continue
            # The request-contract projection passes no verified carrier:
            # the frozen contract, not the withheld candidate, is the
            # authority source there.  A proposer that still names the
            # candidate has written a benign provenance note, not an
            # authority claim — the first probe run abstained on every
            # activated row because this reference was fatal.  Treat it
            # as a non-visible id instead: it grounds nothing, and the
            # node's argument values must be covered by its other cited
            # evidence or the node is still rejected.
            missing_visible_ref = True
            continue
        if evidence_id not in visible_ids:
            missing_visible_ref = True
            continue
        cited_rows.extend(
            row
            for row in visible_evidence
            if str(row.get("event_id") or "").strip() == evidence_id
        )
    if fatal_reasons:
        return tuple(sorted(set(fatal_reasons)))
    if node.kind == "message":
        if missing_visible_ref and not cited_rows:
            return ("required_evidence_not_visible",)
        return ()
    # Citations still have to resolve. Relaxing which values a node may use
    # does not license a node whose provenance names nothing that exists: a
    # graph citing only unresolvable ids has made no claim about the reached
    # state at all, and its arguments are unsupported however common they are.
    if missing_visible_ref and not cited_rows and not allow_carrier:
        return ("required_evidence_not_visible",)
    if not node.arguments:
        return ()
    # The method's grounding rule is that a write argument value is copied
    # verbatim from the visible request or an accepted receipt. Which visible
    # row the proposer names is provenance, not admissibility: judging values
    # only against the rows a node happened to cite rejected values the
    # reached state plainly contains, and an enumerated contract was the only
    # way to escape it. Admissibility is therefore judged against the whole
    # visible evidence set, which forbids fabrication exactly as before.
    allowed_values = _visible_evidence_leaf_value_set(tuple(visible_evidence))
    if allow_carrier:
        allowed_values |= {
            _normalize_argument_leaf(value)
            for call in carrier_tool_calls
            for value in _argument_leaf_values(_mapping_or_none(call.get("arguments")) or {})
        }
    if not allowed_values:
        if missing_visible_ref:
            return ("required_evidence_not_visible",)
        return ("argument_value_not_in_cited_evidence",)
    for key, raw_value in _argument_leaf_items(node.arguments):
        value = _normalize_argument_leaf(raw_value)
        if not value or value in allowed_values:
            continue
        # A closed-vocabulary argument names no object in the environment; its
        # legal values are published by the tool schema itself, identically for
        # every task, so a verbatim hit there grounds it without any private
        # or task-specific source.
        schema_text = parameter_grounding_texts.get(node.tool_name, {}).get(key, "")
        if schema_text and value in schema_text:
            continue
        return ("argument_value_not_in_cited_evidence",)
    return ()


def _node_write_units(
    node: CFRCProposalNode,
    *,
    certificate: CFRCResidualCertificate,
    carrier_tool_calls: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    if node.kind == "message":
        return (MESSAGE_RESIDUAL_UNIT,)
    if node.kind != "tool_call":
        return ()
    matched_units = _matched_residual_obligation_units(
        node,
        certificate=certificate,
        carrier_tool_calls=carrier_tool_calls,
    )
    if matched_units:
        return matched_units
    specific_unit = specific_tool_effect_unit(node.tool_name, node.arguments)
    if specific_unit in certificate.contract_forbidden_units:
        return (specific_unit,)
    if certificate.commitments:
        return (specific_unit,)
    return (tool_effect_unit(node.tool_name),)


def _matched_residual_obligation_units(
    node: CFRCProposalNode,
    *,
    certificate: CFRCResidualCertificate,
    carrier_tool_calls: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    if not certificate.residual_obligations or node.kind != "tool_call":
        return ()
    node_signature = _tool_call_signature(node.tool_name, node.arguments)
    matched: list[str] = []
    node_required_evidence = _canonical_evidence_id_set(node.required_evidence)
    for obligation in certificate.residual_obligations:
        tool_family = tool_effect_unit(node.tool_name)
        if tool_family not in obligation.effect_families:
            continue
        if obligation.provenance_refs and not set(obligation.provenance_refs).issubset(
            node_required_evidence
        ):
            continue
        specific_units = [
            unit
            for unit in obligation.required_unit_tokens
            if unit.startswith(f"{TOOL_CALL_EFFECT_PREFIX}{node.tool_name}:")
        ]
        if specific_units:
            expected = specific_tool_effect_unit(node.tool_name, node.arguments)
            if expected not in specific_units:
                continue
        elif carrier_tool_calls and CHEAP_FRONTIER_CANDIDATE_EVIDENCE in obligation.provenance_refs:
            carrier_signatures = {
                _tool_call_signature(
                    str(call.get("name") or ""),
                    _mapping_or_none(call.get("arguments")) or {},
                )
                for call in carrier_tool_calls
            }
            if node_signature not in carrier_signatures:
                continue
        matched.extend(obligation.required_unit_tokens)
    return tuple(sorted(set(matched)))


def _node_read_units(node: CFRCProposalNode) -> tuple[str, ...]:
    if node.kind != "read_call":
        return ()
    return (read_effect_unit(node.tool_name, node.arguments),)


def _selected_node_ids(
    nodes: Sequence[CFRCProposalNode],
    *,
    node_write_units: Mapping[str, tuple[str, ...]],
    selected_units: set[str],
) -> set[str]:
    by_id = {node.node_id: node for node in nodes}
    selected_ids = {
        node.node_id
        for node in nodes
        if set(node_write_units.get(node.node_id, ())) & selected_units
    }
    frontier = list(selected_ids)
    while frontier:
        node_id = frontier.pop()
        node = by_id.get(node_id)
        if node is None:
            continue
        for dep_id in node.depends_on:
            if dep_id in by_id and dep_id not in selected_ids:
                selected_ids.add(dep_id)
                frontier.append(dep_id)
    return selected_ids


def _selected_obligation_ids(
    nodes: Sequence[CFRCProposalNode],
    *,
    node_write_units: Mapping[str, tuple[str, ...]],
    selected_units: set[str],
    certificate: CFRCResidualCertificate,
) -> set[str]:
    if not certificate.residual_obligations:
        return set()
    selected = set()
    for node in nodes:
        if not (set(node_write_units.get(node.node_id, ())) & selected_units):
            continue
        node_units = set(node_write_units.get(node.node_id, ()))
        for obligation in certificate.residual_obligations:
            if set(obligation.required_unit_tokens) <= node_units:
                selected.add(obligation.obligation_id)
    return selected


def _shared_compiler_frontier(
    *,
    certificate: CFRCResidualCertificate,
    required_carrier_tool_calls: Sequence[Mapping[str, Any]],
    proposal_graph: ProposalGraph,
) -> CFRCFrontier:
    return CFRCFrontier(
        contract=certificate.to_contract(),
        allowed_effect_units=certificate.residual_units
        | frozenset(
            unit for node in proposal_graph.nodes for unit in node.semantic_effects
        ),
        evidence_facts=certificate.evidence_facts,
        frontier_timing=(
            "pre_write" if required_carrier_tool_calls else "post_write"
        ),
        pending_carrier_refs=_pending_carrier_refs(required_carrier_tool_calls),
        executed_commitment_refs=_executed_commitment_carrier_refs(certificate),
        restoration_cost=certificate.restoration_cost_map,
    )


def _shared_compiler_config(
    *,
    certificate: CFRCResidualCertificate,
    nodes: Sequence[CFRCProposalNode],
    compiler_suffix_tool_call_budget: int | None,
    compiler_setting: CompilerSetting,
    require_action_receipt_witness: bool = False,
) -> CFRCCompileConfig:
    base_labels = STRICT_COMPILER_AUTHORITY_LABELS
    tool_call_count = sum(1 for node in nodes if node.kind == "tool_call")
    message_only = any(node.kind == "message" for node in nodes) and tool_call_count == 0
    budget = (
        int(compiler_suffix_tool_call_budget)
        if compiler_suffix_tool_call_budget is not None
        else (0 if message_only else 1)
    )
    normalized_setting = _compiler_setting(compiler_setting)
    lattice_labels = base_labels
    mode = "strict_core"
    collision_budget = 0.0
    if normalized_setting == AUTHORITY_LIFTED_ELL3_COMPILER_SETTING:
        mode = "authority_lifted_extension"
        lattice_labels = tuple(
            dict.fromkeys((*base_labels, ELL3_AUTHORITY_LABEL, ELL_UPPER_AUTHORITY_LABEL))
        )
        collision_budget = max(0.0, float(certificate.collision_budget))
    return CFRCCompileConfig(
        mode=mode,
        authority_lattice=AuthorityLattice.from_ordered_labels(lattice_labels),
        strict_allowed_authorities=frozenset(base_labels),
        allowed_authorities=frozenset(
            label for label in lattice_labels if label != ELL_UPPER_AUTHORITY_LABEL
        ),
        collision_budget=collision_budget,
        suffix_tool_call_budget=max(0, budget),
        require_action_receipt_witness=bool(require_action_receipt_witness),
    )


def _adapter_ell3_execution_guard_errors(
    *,
    certificate: CFRCResidualCertificate,
    compile_result: Any,
) -> tuple[str, ...]:
    if compile_result.selected_authority_label != ELL3_AUTHORITY_LABEL:
        return ()
    projection = compile_result.projection
    readmitted = (
        frozenset(projection.readmitted_footprint)
        if projection is not None
        else frozenset()
    )
    restoration_cost = certificate.restoration_cost_map
    errors: list[str] = []
    if certificate.collision_budget <= 0:
        errors.append("ell3_collision_budget_missing")
    if not restoration_cost:
        errors.append("ell3_restoration_cost_missing")
    if not readmitted:
        errors.append("ell3_committed_unit_readmission_missing")
    elif not readmitted <= frozenset(restoration_cost):
        errors.append("ell3_restoration_cost_missing_for_readmitted_unit")
    return tuple(sorted(set(errors)))


def _shared_compiler_graph_and_bindings(
    *,
    certificate: CFRCResidualCertificate,
    nodes: Sequence[CFRCProposalNode],
    node_write_units: Mapping[str, tuple[str, ...]],
    required_carrier_tool_calls: Sequence[Mapping[str, Any]],
    replica_witnesses: Mapping[str, tuple[str, bool]] | None = None,
    coverage_by_node: Mapping[str, Sequence[str]] | None = None,
) -> tuple[ProposalGraph, dict[str, CFRCNodeBinding]]:
    replica_witnesses = replica_witnesses or {}
    coverage_by_node = coverage_by_node or {}
    compiler_node_ids = {
        node.node_id
        for node in nodes
        if node.kind in {"tool_call", "message"}
    }
    tool_occurrence_counts = Counter(
        node.tool_name for node in nodes if node.kind == "tool_call" and node.tool_name
    )
    graph_nodes: list[ProposalNode] = []
    bindings: dict[str, CFRCNodeBinding] = {}
    evidence_fact_ids = tuple(fact.fact_id for fact in certificate.evidence_facts)
    obligation_units = {
        obligation.obligation_id: tuple(obligation.required_unit_tokens)
        for obligation in certificate.residual_obligations
    }
    for node in nodes:
        if node.kind not in {"tool_call", "message"}:
            continue
        semantic_effects = _shared_compiler_effect_units(
            node,
            node_write_units=node_write_units,
            required_carrier_tool_calls=required_carrier_tool_calls,
            certificate=certificate,
            tool_occurrence_counts=tool_occurrence_counts,
        )
        carrier_refs = _node_carrier_refs(
            node,
            required_carrier_tool_calls=required_carrier_tool_calls,
        )
        closure = build_executable_closure(
            semantic_effects=frozenset(semantic_effects),
            argument_carriers=frozenset(carrier_refs),
            carrier_provenance={
                carrier_ref: "observed_committed_record" for carrier_ref in carrier_refs
            },
        )
        replica_receipt_ref, replica_succeeded = replica_witnesses.get(
            node.node_id,
            ("", None),
        )
        covered_obligation_ids = tuple(
            obligation_id
            for obligation_id in coverage_by_node.get(node.node_id, ())
            if obligation_id in obligation_units
        )
        coverage_claims = tuple(
            ProposalCoverageClaim(
                obligation_id=obligation_id,
                unit_tokens=obligation_units[obligation_id],
                replica_receipt_ref=replica_receipt_ref,
            )
            for obligation_id in covered_obligation_ids
            if replica_receipt_ref
        )
        graph_nodes.append(
            ProposalNode(
                node_id=node.node_id,
                kind=("tool_call" if node.kind == "tool_call" else "message"),
                authority_label=str(node.authority_level or "ell_write").strip(),
                semantic_effects=semantic_effects,
                effect_families=tuple(sorted({tool_effect_unit(node.tool_name)}))
                if node.kind == "tool_call"
                else (MESSAGE_RESIDUAL_UNIT,),
                dependency_ids=tuple(
                    dep_id for dep_id in node.depends_on if dep_id in compiler_node_ids
                ),
                provenance_refs=tuple(
                    evidence_id
                    for evidence_id in _canonical_required_evidence(node)
                    if evidence_id != CHEAP_FRONTIER_CANDIDATE_EVIDENCE
                ),
                evidence_fact_ids=tuple(
                    evidence_id
                    for evidence_id in _canonical_required_evidence(node)
                    if evidence_id != CHEAP_FRONTIER_CANDIDATE_EVIDENCE
                ),
                coverage_claims=coverage_claims,
                executable_closure=closure,
                tool_call_count=1 if node.kind == "tool_call" else 0,
            )
        )
        bindings[node.node_id] = CFRCNodeBinding(
            node_id=node.node_id,
            executable_closure=closure,
            suffix_payload=(
                {"name": node.tool_name, "arguments": dict(node.arguments)}
                if node.kind == "tool_call"
                else {"content": node.message}
            ),
            binding_kind=("tool_call" if node.kind == "tool_call" else "message"),
            carrier_refs=carrier_refs,
            replica_receipt_ref=replica_receipt_ref,
            replica_execution_succeeded=replica_succeeded,
        )
    graph = ProposalGraph(
        nodes=tuple(graph_nodes),
        graph_id=f"tau2_{certificate.domain}_proposal_graph",
        commitment_ids=tuple(
            commitment.commitment_id for commitment in certificate.commitments
        ),
        residual_obligation_ids=tuple(
            obligation.obligation_id for obligation in certificate.residual_obligations
        ),
        evidence_fact_ids=evidence_fact_ids,
    )
    return graph, bindings


def _shared_compiler_effect_units(
    node: CFRCProposalNode,
    *,
    node_write_units: Mapping[str, tuple[str, ...]],
    required_carrier_tool_calls: Sequence[Mapping[str, Any]],
    certificate: CFRCResidualCertificate,
    tool_occurrence_counts: Mapping[str, int],
) -> tuple[str, ...]:
    base_units = tuple(sorted(node_write_units.get(node.node_id, ())))
    if node.kind != "tool_call":
        return base_units
    if base_units != (tool_effect_unit(node.tool_name),):
        return base_units
    if (
        tool_occurrence_counts.get(node.tool_name, 0) > 1
        or required_carrier_tool_calls
        or certificate.commitments
    ):
        return (specific_tool_effect_unit(node.tool_name, node.arguments),)
    return base_units


def _pending_carrier_refs(
    tool_calls: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    refs: list[str] = []
    for index, call in enumerate(tool_calls):
        refs.append(_carrier_ref_from_tool_call(call, phase="pending", index=index))
    return tuple(refs)


def _executed_commitment_carrier_refs(
    certificate: CFRCResidualCertificate,
) -> tuple[str, ...]:
    refs: list[str] = []
    for index, commitment in enumerate(certificate.commitments):
        refs.append(
            f"carrier.executed:{_stable_public_hash(commitment.commitment_id)}:{index}"
        )
    return tuple(refs)


def _node_carrier_refs(
    node: CFRCProposalNode,
    *,
    required_carrier_tool_calls: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    if node.kind != "tool_call":
        return ()
    if (
        CHEAP_FRONTIER_CANDIDATE_EVIDENCE
        not in _canonical_evidence_id_set(node.required_evidence)
    ):
        return ()
    refs: list[str] = []
    for index, call in enumerate(required_carrier_tool_calls):
        call_name = str(call.get("name") or "").strip()
        if call_name != node.tool_name:
            continue
        refs.append(_carrier_ref_from_tool_call(call, phase="pending", index=index))
    return tuple(dict.fromkeys(refs))


def _carrier_ref_from_tool_call(
    call: Mapping[str, Any],
    *,
    phase: Literal["pending", "executed"],
    index: int,
) -> str:
    tool_name = str(call.get("name") or "").strip()
    arguments = _mapping_or_none(call.get("arguments")) or {}
    signature_hash = _stable_public_hash(_tool_call_signature(tool_name, arguments))
    return f"carrier.{phase}:{signature_hash}:{index}"


def _selected_node_ids_with_terminal_completion(
    nodes: Sequence[CFRCProposalNode],
    *,
    compiler_selected_ids: set[str],
    include_message_dependents: bool = False,
) -> set[str]:
    by_id = {node.node_id: node for node in nodes}
    selected_ids = set(compiler_selected_ids)
    frontier = list(compiler_selected_ids)
    while frontier:
        node_id = frontier.pop()
        node = by_id.get(node_id)
        if node is None:
            continue
        for dep_id in node.depends_on:
            if dep_id in by_id and dep_id not in selected_ids:
                selected_ids.add(dep_id)
                frontier.append(dep_id)
    if include_message_dependents:
        for node in sorted(nodes, key=lambda item: item.node_id):
            if node.kind != "message" or not node.depends_on:
                continue
            if set(node.depends_on) <= selected_ids:
                selected_ids.add(node.node_id)
    return selected_ids


def _cited_visible_evidence_rows(
    required_evidence: Sequence[str],
    visible_evidence: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    cited = _canonical_evidence_id_set(required_evidence)
    return tuple(
        row
        for row in visible_evidence
        if str(row.get("event_id") or "").strip() in cited
    )


def _canonical_required_evidence(node: CFRCProposalNode) -> tuple[str, ...]:
    return tuple(
        evidence_id
        for raw_id in node.required_evidence
        if (evidence_id := _canonical_evidence_id(raw_id))
    )


def _canonical_evidence_id_set(required_evidence: Sequence[str]) -> set[str]:
    return {
        evidence_id
        for raw_id in required_evidence
        if (evidence_id := _canonical_evidence_id(raw_id))
    }


def _canonical_evidence_id(raw_id: Any) -> str:
    evidence_id = str(raw_id or "").strip()
    if not evidence_id:
        return ""
    if evidence_id.startswith("visible_evidence:"):
        return evidence_id.split(":", 1)[1].strip()
    if evidence_id.startswith(f"{CHEAP_FRONTIER_CANDIDATE_EVIDENCE}:"):
        return CHEAP_FRONTIER_CANDIDATE_EVIDENCE
    return evidence_id


def _argument_leaf_values(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Mapping):
        leaves: list[Any] = []
        for item in value.values():
            leaves.extend(_argument_leaf_values(item))
        return tuple(leaves)
    if isinstance(value, (list, tuple)):
        leaves = []
        for item in value:
            leaves.extend(_argument_leaf_values(item))
        return tuple(leaves)
    if value is None:
        return ()
    return (value,)


def _visible_evidence_leaf_value_set(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    values: set[str] = set()
    for row in rows:
        for value in _argument_leaf_values(_row_with_parsed_content(row)):
            normalized = _normalize_argument_leaf(value)
            if normalized:
                values.add(normalized)
    return values


def _row_with_parsed_content(row: Mapping[str, Any]) -> Mapping[str, Any]:
    content = row.get("content")
    if not isinstance(content, str):
        return row
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return row
    if not isinstance(parsed, (Mapping, list)):
        return row
    payload = dict(row)
    payload["content"] = parsed
    return payload


def _normalize_argument_leaf(value: Any) -> str:
    return str(value).strip()


def _carrier_signature_counts(
    tool_calls: Sequence[Mapping[str, Any]],
) -> Counter[str]:
    return Counter(
        _tool_call_signature(
            str(call.get("name") or ""),
            _mapping_or_none(call.get("arguments")) or {},
        )
        for call in tool_calls
        if str(call.get("name") or "").strip()
    )


def _tool_call_signature(tool_name: str, arguments: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            "name": str(tool_name).strip(),
            "arguments": canonical_call_arguments(dict(arguments)),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _receipt_bound_commitments(
    tool_calls: Sequence[Mapping[str, Any]],
    tool_receipts: Sequence[Any],
) -> list[Commitment]:
    receipts_by_id = {
        receipt_id: receipt
        for receipt in tool_receipts
        if (
            receipt_id := (
                _message_attr_str(receipt, "tool_call_id")
                or _message_attr_str(receipt, "id")
            )
        )
    }
    commitments: list[Commitment] = []
    for call in tool_calls:
        call_id = str(call.get("tool_call_id") or call.get("id") or "").strip()
        if not call_id:
            continue
        receipt = receipts_by_id.get(call_id)
        if receipt is None:
            continue
        if not _tool_receipt_status(receipt)["success"]:
            continue
        tool_name = str(call.get("name") or "").strip()
        arguments = _mapping_or_none(call.get("arguments")) or {}
        unit = specific_tool_effect_unit(tool_name, arguments)
        commitments.append(
            Commitment(
                commitment_id=f"receipt:{tool_name}:{_stable_public_hash(_tool_call_signature(tool_name, arguments))}",
                unit_tokens=(unit,),
                effect_families=(tool_effect_unit(tool_name),),
                provenance_refs=("receipt_success",),
                authority_label="ell_write",
            )
        )
    return commitments


def _frontier_candidate_residual_obligations(
    tool_calls: Sequence[Mapping[str, Any]],
) -> list[ResidualObligation]:
    obligations: list[ResidualObligation] = []
    for index, call in enumerate(tool_calls):
        tool_name = str(call.get("name") or "").strip()
        arguments = _mapping_or_none(call.get("arguments")) or {}
        obligations.append(
            ResidualObligation(
                obligation_id=f"frontier:{tool_name}:{index}",
                required_unit_tokens=(specific_tool_effect_unit(tool_name, arguments),),
                effect_families=(tool_effect_unit(tool_name),),
                provenance_refs=(CHEAP_FRONTIER_CANDIDATE_EVIDENCE,),
                full_discharge_required=True,
            )
        )
    return obligations


def _open_residual_obligations_after_commitments(
    obligations: Sequence[ResidualObligation],
    commitments: Sequence[Commitment],
) -> tuple[ResidualObligation, ...]:
    committed_units = {
        unit for commitment in commitments for unit in commitment.unit_tokens
    }
    if not committed_units:
        return tuple(obligations)
    return tuple(
        obligation
        for obligation in obligations
        if not (set(obligation.required_unit_tokens) & committed_units)
    )


def _assert_no_explicit_k_r_overlap(certificate: CFRCResidualCertificate) -> None:
    if certificate.explicit_k_r_overlap_units:
        raise AssertionError("cfrc_certificate_explicit_k_r_overlap")


def _evidence_facts_from_visible_rows(
    visible_rows: Sequence[Mapping[str, Any]],
) -> list[EvidenceFact]:
    facts: list[EvidenceFact] = []
    for row in visible_rows:
        event_id = str(row.get("event_id") or "").strip()
        if not event_id:
            continue
        facts.append(
            EvidenceFact(
                fact_id=event_id,
                fact_type=str(row.get("kind") or "visible_evidence"),
                provenance_refs=(event_id,),
            )
        )
    return facts


def _tool_receipt_status(message: Any) -> dict[str, bool]:
    native_error = _message_attr(message, "error")
    if isinstance(native_error, bool):
        return {
            "success": not native_error,
            "error": native_error,
            "unknown": False,
        }
    raw_data = _mapping_or_none(getattr(message, "raw_data", None)) or {}
    payloads = [
        _mapping_or_none(raw_data.get("papc_cfrc")),
        raw_data,
        _mapping_or_none(getattr(message, "__dict__", None)),
    ]
    for payload in payloads:
        if not payload:
            continue
        status = str(payload.get("status") or "").strip().lower()
        if status in {"success", "ok", "completed"}:
            return {"success": True, "error": False, "unknown": False}
        if status in {"error", "failed", "exception"}:
            return {"success": False, "error": True, "unknown": False}
        if payload.get("success") is True or payload.get("ok") is True:
            return {"success": True, "error": False, "unknown": False}
        if payload.get("success") is False or payload.get("error") is True:
            return {"success": False, "error": True, "unknown": False}
    text = str(getattr(message, "content", "") or "").strip().lower()
    if text:
        if any(token in text for token in ("error", "failed", "exception", "invalid")):
            return {"success": False, "error": True, "unknown": False}
        if any(token in text for token in ("success", "completed", "updated", "done")):
            return {"success": True, "error": False, "unknown": False}
        if _looks_like_successful_structured_receipt(text):
            return {"success": True, "error": False, "unknown": False}
    return {"success": False, "error": False, "unknown": True}


def _receipt_content_value(value: Any) -> Any:
    if not isinstance(value, str):
        return json_safe_value(value)
    try:
        return json_safe_value(json.loads(value))
    except json.JSONDecodeError:
        return value.strip()


def _request_contract_receipt_text(value: str) -> str:
    """Serialize structured receipts compactly without dropping any fields."""

    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return value.strip()
    if isinstance(payload, Mapping) and isinstance(
        payload.get("variants"), Mapping
    ):
        variants = payload["variants"]
        if all(isinstance(item, Mapping) for item in variants.values()):
            variant_field_names: list[str] = []
            option_field_names: list[str] = []
            for raw_variant in variants.values():
                assert isinstance(raw_variant, Mapping)
                for field_name in raw_variant:
                    normalized = str(field_name)
                    if (
                        normalized not in {"item_id", "options"}
                        and normalized not in variant_field_names
                    ):
                        variant_field_names.append(normalized)
                options = raw_variant.get("options")
                if isinstance(options, Mapping):
                    for option_name in options:
                        normalized = str(option_name)
                        if normalized not in option_field_names:
                            option_field_names.append(normalized)
            projected = {
                key: json_safe_value(item)
                for key, item in payload.items()
                if key != "variants"
            }
            projected["projection_schema"] = (
                "papc.tau2.product_receipt_table.v1"
            )
            projected["variant_fields"] = [
                "item_id",
                *variant_field_names,
                *[f"options.{name}" for name in option_field_names],
            ]
            projected["variants"] = [
                [
                    str(raw_variant.get("item_id") or variant_id),
                    *[
                        json_safe_value(raw_variant.get(field_name))
                        for field_name in variant_field_names
                    ],
                    *[
                        json_safe_value(
                            (
                                raw_variant.get("options")
                                if isinstance(
                                    raw_variant.get("options"), Mapping
                                )
                                else {}
                            ).get(option_name)
                        )
                        for option_name in option_field_names
                    ],
                ]
                for variant_id, raw_variant in variants.items()
                if isinstance(raw_variant, Mapping)
            ]
            payload = projected
    return json.dumps(
        json_safe_value(payload),
        ensure_ascii=False,
        separators=(",", ":"),
    )


_ESCALATION_TOKENS = frozenset(
    {
        "human",
        "agent",
        "person",
        "representative",
        "supervisor",
        "manager",
        "escalate",
        "escalation",
        "manual",
    }
)


def _visible_user_requests_escalation(messages: Sequence[Any]) -> bool:
    """Whether any visible user turn asks for a person by name.

    The token family is generic customer-service vocabulary, not task
    content; a give-up transfer with no such request is the cheap agent
    abandoning a completable request, which is exactly the residual the
    successor exists to complete.
    """

    for message in messages:
        if _message_attr_str(message, "role") != "user":
            continue
        text = str(_message_attr(message, "content") or "").casefold()
        tokens = set(re.findall(r"[a-z]+", text))
        if tokens & _ESCALATION_TOKENS:
            return True
    return False


def _derived_terminal_literals(
    derived_rows: Sequence[Mapping[str, Any]],
    candidate_calls: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Select the receipt-arithmetic values the terminal must report.

    Every value is deterministic receipt arithmetic already emitted as
    derivation evidence; a literal attaches only when its derivation row
    matches a withheld candidate write exactly, so the terminal obligation
    never exceeds what the receipts determine.
    """

    cancel_orders: set[str] = set()
    return_orders: set[str] = set()
    pair_targets: set[tuple[str, str, str]] = set()
    for call in candidate_calls:
        arguments = call.get("arguments")
        if not isinstance(arguments, Mapping):
            continue
        name = str(call.get("name") or "")
        order_id = str(arguments.get("order_id") or "").strip()
        if name == "cancel_pending_order" and order_id:
            cancel_orders.add(order_id)
        if name == "return_delivered_order_items" and order_id:
            return_orders.add(order_id)
        if name in (
            "exchange_delivered_order_items",
            "modify_pending_order_items",
        ):
            item_ids = [
                str(item or "").strip()
                for item in _list_value(arguments.get("item_ids"))
            ]
            new_item_ids = [
                str(item or "").strip()
                for item in _list_value(arguments.get("new_item_ids"))
            ]
            for current_item, new_item in zip(item_ids, new_item_ids):
                if order_id and current_item and new_item:
                    pair_targets.add((order_id, current_item, new_item))
    literals: list[str] = []
    for row in derived_rows:
        content = row.get("content")
        if not isinstance(content, Mapping):
            continue
        schema = str(content.get("schema") or "")
        order_id = str(content.get("order_id") or "").strip()
        value = ""
        if (
            schema == "papc.tau2.retail_cancellation_total_evidence.v1"
            and order_id in cancel_orders
        ):
            value = str(content.get("total_paid_display") or "")
        elif (
            schema == "papc.tau2.retail_return_total_evidence.v1"
            and order_id in return_orders
        ):
            value = str(content.get("total_refund_display") or "")
        elif schema == "papc.tau2.retail_price_delta_evidence.v1":
            pair = (
                order_id,
                str(content.get("current_item_id") or "").strip(),
                str(content.get("candidate_new_item_id") or "").strip(),
            )
            if pair in pair_targets:
                value = str(content.get("price_difference_display") or "")
        if value and value not in literals:
            literals.append(value)
    return literals


def _retail_request_contract_derived_evidence(
    *,
    messages: Sequence[Any],
    candidate_calls: Sequence[Mapping[str, Any]],
    target_source: Literal[
        "untrusted_candidate",
        "frozen_request_contract",
    ] = "untrusted_candidate",
) -> list[dict[str, Any]]:
    """Derive exact price deltas from receipts without granting target authority."""

    candidate_new_item_ids: set[str] = set()
    modification_candidates: list[dict[str, Any]] = []
    return_candidates: list[dict[str, Any]] = []
    cancellation_candidates: list[dict[str, Any]] = []
    for call in candidate_calls:
        arguments = call.get("arguments")
        if not isinstance(arguments, Mapping):
            continue
        call_name = str(call.get("name") or "")
        if call_name == "return_delivered_order_items":
            order_id = str(arguments.get("order_id") or "").strip()
            item_ids = [
                str(item_id or "").strip()
                for item_id in _list_value(arguments.get("item_ids"))
                if str(item_id or "").strip()
            ]
            payment_method_id = str(
                arguments.get("payment_method_id") or ""
            ).strip()
            if order_id and item_ids:
                return_candidates.append(
                    {
                        "order_id": order_id,
                        "item_ids": item_ids,
                        "payment_method_id": payment_method_id,
                    }
                )
        if call_name == "cancel_pending_order":
            order_id = str(arguments.get("order_id") or "").strip()
            if order_id:
                cancellation_candidates.append({"order_id": order_id})
        if call_name == "modify_pending_order_items":
            order_id = str(arguments.get("order_id") or "").strip()
            item_ids = [
                str(item_id or "").strip()
                for item_id in _list_value(arguments.get("item_ids"))
                if str(item_id or "").strip()
            ]
            new_item_ids = [
                str(item_id or "").strip()
                for item_id in _list_value(arguments.get("new_item_ids"))
                if str(item_id or "").strip()
            ]
            payment_method_id = str(
                arguments.get("payment_method_id") or ""
            ).strip()
            if order_id and item_ids and len(item_ids) == len(new_item_ids):
                modification_candidates.append(
                    {
                        "order_id": order_id,
                        "item_ids": item_ids,
                        "new_item_ids": new_item_ids,
                        "payment_method_id": payment_method_id,
                    }
                )
        raw_item_ids = arguments.get("new_item_ids")
        if not isinstance(raw_item_ids, list):
            continue
        candidate_new_item_ids.update(
            str(item_id or "").strip()
            for item_id in raw_item_ids
            if str(item_id or "").strip()
        )

    receipts: list[Mapping[str, Any]] = []
    for message in messages:
        if (
            _message_attr_str(message, "role") != "tool"
            and not _message_attr_str(message, "tool_call_id")
        ):
            continue
        payload = _receipt_content_value(_message_attr(message, "content"))
        if isinstance(payload, Mapping):
            receipts.append(payload)

    rows: list[dict[str, Any]] = []
    order_receipts = {
        str(receipt.get("order_id") or "").strip(): receipt
        for receipt in receipts
        if str(receipt.get("order_id") or "").strip()
    }
    for candidate in return_candidates:
        order_id = str(candidate["order_id"])
        item_ids = list(candidate["item_ids"])
        receipt = order_receipts.get(order_id)
        receipt_items = (
            receipt.get("items")
            if isinstance(receipt, Mapping)
            else None
        )
        if not isinstance(receipt_items, list):
            continue
        price_by_item_id = {
            str(item.get("item_id") or "").strip(): _money_decimal(
                item.get("price")
            )
            for item in receipt_items
            if isinstance(item, Mapping)
        }
        if any(
            not item_id or price_by_item_id.get(item_id) is None
            for item_id in item_ids
        ):
            continue
        item_prices = [
            {
                "item_id": item_id,
                "price_usd": format(price_by_item_id[item_id], ".2f"),
            }
            for item_id in item_ids
        ]
        total_refund = sum(
            (price_by_item_id[item_id] for item_id in item_ids),
            Decimal("0.00"),
        )
        rows.append(
            {
                "event_id": f"derived_return_total_{len(rows)}",
                "kind": "deterministic_receipt_derivation",
                "content": {
                    "schema": (
                        "papc.tau2.retail_return_total_evidence.v1"
                    ),
                    "order_id": order_id,
                    "item_prices": item_prices,
                    "total_refund_usd": format(total_refund, ".2f"),
                    "total_refund_display": (
                        f"${format(total_refund, '.2f')}"
                    ),
                    "payment_method_id": candidate["payment_method_id"],
                    **(
                        {"target_source": "frozen_request_contract"}
                        if target_source == "frozen_request_contract"
                        else {"candidate_target_is_untrusted": True}
                    ),
                    "authority": "arithmetic_evidence_only",
                },
            }
        )

    # Cancelling a pending order refunds the amounts its payment history
    # actually charged; that total is the same deterministic receipt
    # arithmetic as a return total, and without it every cancellation
    # refund literal has no carrier.
    for candidate in cancellation_candidates:
        order_id = str(candidate["order_id"])
        receipt = order_receipts.get(order_id)
        if not isinstance(receipt, Mapping):
            continue
        payments = [
            entry
            for entry in (
                receipt.get("payment_history")
                if isinstance(receipt.get("payment_history"), list)
                else []
            )
            if isinstance(entry, Mapping)
            and str(entry.get("transaction_type") or "").casefold()
            == "payment"
        ]
        amounts = [_money_decimal(entry.get("amount")) for entry in payments]
        if not amounts or any(amount is None for amount in amounts):
            continue
        total_paid = sum(amounts, Decimal("0.00"))
        rows.append(
            {
                "event_id": f"derived_cancellation_total_{len(rows)}",
                "kind": "deterministic_receipt_derivation",
                "content": {
                    "schema": (
                        "papc.tau2.retail_cancellation_total_evidence.v1"
                    ),
                    "order_id": order_id,
                    "payment_amounts_usd": [
                        format(amount, ".2f") for amount in amounts
                    ],
                    "total_paid_usd": format(total_paid, ".2f"),
                    "total_paid_display": f"${format(total_paid, '.2f')}",
                    "payment_method_ids": sorted(
                        {
                            str(entry.get("payment_method_id") or "").strip()
                            for entry in payments
                            if str(entry.get("payment_method_id") or "").strip()
                        }
                    ),
                    **(
                        {"target_source": "frozen_request_contract"}
                        if target_source == "frozen_request_contract"
                        else {"candidate_target_is_untrusted": True}
                    ),
                    "authority": "arithmetic_evidence_only",
                },
            }
        )

    for order_id, receipt in order_receipts.items():
        receipt_items = receipt.get("items")
        returned_item_ids = sorted(
            {
                str(item_id or "").strip()
                for item_id in _list_value(receipt.get("return_items"))
                if str(item_id or "").strip()
            }
        )
        if not returned_item_ids or not isinstance(receipt_items, list):
            continue
        price_by_item_id = {
            str(item.get("item_id") or "").strip(): _money_decimal(
                item.get("price")
            )
            for item in receipt_items
            if isinstance(item, Mapping)
        }
        if any(
            price_by_item_id.get(item_id) is None
            for item_id in returned_item_ids
        ):
            continue
        returned = set(returned_item_ids)
        returned_total = sum(
            (
                price
                for item_id, price in price_by_item_id.items()
                if item_id in returned and price is not None
            ),
            Decimal("0.00"),
        )
        remaining_total = sum(
            (
                price
                for item_id, price in price_by_item_id.items()
                if item_id not in returned and price is not None
            ),
            Decimal("0.00"),
        )
        rows.append(
            {
                "event_id": f"derived_return_state_total_{len(rows)}",
                "kind": "deterministic_receipt_derivation",
                "content": {
                    "schema": (
                        "papc.tau2.retail_return_state_total_evidence.v1"
                    ),
                    "order_id": order_id,
                    "returned_item_ids": returned_item_ids,
                    "returned_item_total_usd": format(
                        returned_total, ".2f"
                    ),
                    "returned_item_total_display": (
                        f"${format(returned_total, '.2f')}"
                    ),
                    "remaining_item_total_usd": format(
                        remaining_total, ".2f"
                    ),
                    "remaining_item_total_display": (
                        f"${format(remaining_total, '.2f')}"
                    ),
                    "authority": "arithmetic_evidence_only",
                },
            }
        )

    if not candidate_new_item_ids:
        return rows

    target_variants: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for receipt in receipts:
        product_id = str(receipt.get("product_id") or "").strip()
        variants = receipt.get("variants")
        if not product_id or not isinstance(variants, Mapping):
            continue
        for variant_id, raw_variant in variants.items():
            if not isinstance(raw_variant, Mapping):
                continue
            item_id = str(
                raw_variant.get("item_id") or variant_id or ""
            ).strip()
            if item_id in candidate_new_item_ids:
                target_variants[item_id] = (product_id, raw_variant)

    price_difference_by_item_pair: dict[tuple[str, str, str], Decimal] = {}
    for receipt in receipts:
        order_id = str(receipt.get("order_id") or "").strip()
        status = str(receipt.get("status") or "").strip().casefold()
        items = receipt.get("items")
        # Pending orders take item modifications and delivered orders take
        # item exchanges; the price difference between the current item in
        # the order receipt and the target variant in the product receipt
        # is the same deterministic arithmetic in both cases.  The earlier
        # pending-only filter silenced every exchange delta, which left the
        # verifier's exact difference literals without a carrier.
        if (
            not order_id
            or not (
                status.startswith("pending") or status.startswith("delivered")
            )
            or not isinstance(items, list)
        ):
            continue
        payment_history = receipt.get("payment_history")
        payment_method_ids = sorted(
            {
                str(entry.get("payment_method_id") or "").strip()
                for entry in (
                    payment_history if isinstance(payment_history, list) else []
                )
                if isinstance(entry, Mapping)
                and str(entry.get("payment_method_id") or "").strip()
            }
        )
        for raw_item in items:
            if not isinstance(raw_item, Mapping):
                continue
            current_item_id = str(raw_item.get("item_id") or "").strip()
            current_product_id = str(
                raw_item.get("product_id") or ""
            ).strip()
            current_price = _money_decimal(raw_item.get("price"))
            if not current_item_id or current_price is None:
                continue
            for new_item_id, (product_id, variant) in target_variants.items():
                if current_product_id != product_id:
                    continue
                new_price = _money_decimal(variant.get("price"))
                if new_price is None:
                    continue
                difference = new_price - current_price
                price_difference_by_item_pair[
                    (order_id, current_item_id, new_item_id)
                ] = difference
                difference_sign = (
                    "+" if difference > 0 else "-" if difference < 0 else ""
                )
                rows.append(
                    {
                        "event_id": f"derived_price_delta_{len(rows)}",
                        "kind": "deterministic_receipt_derivation",
                        "content": {
                            "schema": (
                                "papc.tau2.retail_price_delta_evidence.v1"
                            ),
                            "order_id": order_id,
                            "current_item_id": current_item_id,
                            "candidate_new_item_id": new_item_id,
                            "current_price_usd": format(current_price, ".2f"),
                            "candidate_price_usd": format(new_price, ".2f"),
                            "price_difference_usd": format(difference, ".2f"),
                            "price_difference_display": (
                                f"{difference_sign}"
                                f"${format(abs(difference), '.2f')}"
                            ),
                            "existing_payment_method_ids": payment_method_ids,
                            **(
                                {"target_source": "frozen_request_contract"}
                                if target_source == "frozen_request_contract"
                                else {"candidate_target_is_untrusted": True}
                            ),
                            "authority": "arithmetic_evidence_only",
                        },
                    }
                )

    payment_balance_by_id: dict[str, Decimal] = {}
    for receipt in receipts:
        payment_methods = receipt.get("payment_methods")
        if not isinstance(payment_methods, Mapping):
            continue
        for raw_id, raw_payment_method in payment_methods.items():
            if not isinstance(raw_payment_method, Mapping):
                continue
            payment_method_id = str(
                raw_payment_method.get("id") or raw_id or ""
            ).strip()
            balance = _money_decimal(raw_payment_method.get("balance"))
            if payment_method_id and balance is not None:
                payment_balance_by_id[payment_method_id] = balance

    for candidate in modification_candidates:
        order_id = str(candidate["order_id"])
        differences = [
            price_difference_by_item_pair.get(
                (order_id, str(current_item_id), str(new_item_id))
            )
            for current_item_id, new_item_id in zip(
                candidate["item_ids"],
                candidate["new_item_ids"],
                strict=True,
            )
        ]
        if not differences or any(item is None for item in differences):
            continue
        total = sum(
            (item for item in differences if item is not None),
            Decimal("0.00"),
        )
        sign = "+" if total > 0 else "-" if total < 0 else ""
        payment_method_id = str(candidate["payment_method_id"])
        content: dict[str, Any] = {
            "schema": "papc.tau2.retail_modification_total_evidence.v1",
            "order_id": order_id,
            "total_price_difference_usd": format(total, ".2f"),
            "total_price_difference_display": (
                f"{sign}${format(abs(total), '.2f')}"
            ),
            "total_price_difference_abs_display": (
                f"${format(abs(total), '.2f')}"
            ),
            "settlement_direction": (
                "additional_charge"
                if total > 0
                else "refund"
                if total < 0
                else "no_change"
            ),
            "payment_method_id": payment_method_id,
            **(
                {"target_source": "frozen_request_contract"}
                if target_source == "frozen_request_contract"
                else {"candidate_target_is_untrusted": True}
            ),
            "authority": "arithmetic_evidence_only",
        }
        balance_before = payment_balance_by_id.get(payment_method_id)
        if balance_before is not None:
            balance_after = balance_before - total
            content["payment_balance_after_usd"] = format(
                balance_after, ".2f"
            )
            content["payment_balance_after_display"] = (
                f"${format(balance_after, '.2f')}"
            )
        rows.append(
            {
                "event_id": f"derived_modification_total_{len(rows)}",
                "kind": "deterministic_receipt_derivation",
                "content": content,
            }
        )
    return rows


def _money_decimal(value: Any) -> Decimal | None:
    try:
        return Decimal(str(value)).quantize(Decimal("0.01"))
    except (InvalidOperation, TypeError, ValueError):
        return None


def _looks_like_successful_structured_receipt(text: str) -> bool:
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return False
    if isinstance(payload, Mapping):
        return bool(payload)
    return False


def _tool_receipt_summary(tool_receipts: Sequence[Any]) -> dict[str, Any]:
    success = 0
    error = 0
    unknown = 0
    for receipt in tool_receipts:
        status = _tool_receipt_status(receipt)
        success += int(status["success"])
        error += int(status["error"])
        unknown += int(status["unknown"])
    return {
        "success_count": success,
        "error_count": error,
        "unknown_count": unknown,
        "all_succeeded": bool(tool_receipts) and error == 0 and unknown == 0,
        "has_error": error > 0,
        "has_unknown": unknown > 0,
    }


def _suffix_final_message_from_receipts(
    tool_receipts: Sequence[Any],
) -> tuple[str, dict[str, Any]]:
    summary = _tool_receipt_summary(tool_receipts)
    if summary["all_succeeded"]:
        message = (
            "I completed the remaining requested update after successful tool receipts."
        )
    elif summary["has_error"]:
        message = (
            "I attempted the projected residual action, but at least one tool receipt "
            "reported an error, so I cannot confirm completion."
        )
    else:
        message = (
            "I issued the projected residual action, but the tool receipts did not "
            "clearly confirm success."
        )
    return message, summary


def receipt_bound_commitments(
    tool_calls: Sequence[Mapping[str, Any]],
    tool_receipts: Sequence[Any],
) -> list[Commitment]:
    return _receipt_bound_commitments(tool_calls, tool_receipts)


def frontier_candidate_residual_obligations(
    tool_calls: Sequence[Mapping[str, Any]],
) -> list[ResidualObligation]:
    return _frontier_candidate_residual_obligations(tool_calls)


def open_residual_obligations_after_commitments(
    obligations: Sequence[ResidualObligation],
    commitments: Sequence[Commitment],
) -> tuple[ResidualObligation, ...]:
    """Drop residual obligations whose units already entered K_exec.

    A successful receipt commits its unit into ``K_exec``; that unit must not
    remain an explicit residual obligation, otherwise the certificate would carry
    a K/R overlap and re-propose an already-committed effect.
    """

    return _open_residual_obligations_after_commitments(obligations, commitments)


def evidence_facts_from_visible_rows(
    visible_rows: Sequence[Mapping[str, Any]],
) -> list[EvidenceFact]:
    return _evidence_facts_from_visible_rows(visible_rows)


def suffix_final_message_from_receipts(
    tool_receipts: Sequence[Any],
) -> tuple[str, dict[str, Any]]:
    return _suffix_final_message_from_receipts(tool_receipts)


def _abstain_bucket(
    *,
    proposal: CFRCProposalGraph,
    valid_typed_node_count: int,
    reject_counts: Mapping[str, int],
) -> str:
    if not proposal.nodes:
        return "strong_proposed_no_tool_nodes"
    if valid_typed_node_count > 0:
        return "valid_typed_nodes_before_pi"
    if reject_counts:
        return "invalid_typed_nodes"
    return "unknown"


def _toposort_nodes(nodes: Sequence[CFRCProposalNode]) -> tuple[CFRCProposalNode, ...]:
    by_id = {node.node_id: node for node in nodes}
    selected: list[CFRCProposalNode] = []
    temporary: set[str] = set()
    permanent: set[str] = set()

    def visit(node: CFRCProposalNode) -> None:
        if node.node_id in permanent:
            return
        if node.node_id in temporary:
            return
        temporary.add(node.node_id)
        for dep in node.depends_on:
            if dep in by_id:
                visit(by_id[dep])
        temporary.remove(node.node_id)
        permanent.add(node.node_id)
        selected.append(node)

    for node in nodes:
        visit(node)
    return tuple(selected)


def _abstain(
    certificate: CFRCResidualCertificate,
    proposal: CFRCProposalGraph,
    *,
    reason: str,
    projection: Mapping[str, Any] | None = None,
) -> CFRCProjectionResult:
    return CFRCProjectionResult(
        execute=False,
        selected_nodes=(),
        certificate=certificate,
        proposal=proposal,
        projection=projection or {"decision": "project_abstain"},
        reason=reason,
    )


def _json_object_from_text(text: str) -> dict[str, Any]:
    stripped = str(text or "").strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("json_object_not_found") from None
        try:
            payload = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"json_decode_error:{exc.msg}") from None
    if not isinstance(payload, dict):
        raise ValueError("json_payload_not_object")
    return payload


def _list_value(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _mapping_or_none(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _coerce_commitment(value: Commitment | Mapping[str, Any]) -> Commitment:
    if isinstance(value, Commitment):
        return value
    payload = dict(value)
    return Commitment(
        commitment_id=str(payload.get("commitment_id") or "receipt_commitment"),
        unit_tokens=tuple(
            str(item).strip() for item in _list_value(payload.get("unit_tokens")) if str(item).strip()
        ),
        effect_families=tuple(
            str(item).strip()
            for item in _list_value(payload.get("effect_families"))
            if str(item).strip()
        ),
        provenance_refs=tuple(
            str(item).strip()
            for item in _list_value(payload.get("provenance_refs"))
            if str(item).strip()
        ),
        authority_label=str(payload.get("authority_label") or "").strip(),
    )


def _coerce_residual_obligation(
    value: ResidualObligation | Mapping[str, Any],
) -> ResidualObligation:
    if isinstance(value, ResidualObligation):
        return value
    payload = dict(value)
    return ResidualObligation(
        obligation_id=str(payload.get("obligation_id") or "residual_obligation"),
        required_unit_tokens=tuple(
            str(item).strip()
            for item in _list_value(payload.get("required_unit_tokens"))
            if str(item).strip()
        ),
        effect_families=tuple(
            str(item).strip()
            for item in _list_value(payload.get("effect_families"))
            if str(item).strip()
        ),
        provenance_refs=tuple(
            str(item).strip()
            for item in _list_value(payload.get("provenance_refs"))
            if str(item).strip()
        ),
        full_discharge_required=bool(payload.get("full_discharge_required", True)),
    )


def _coerce_evidence_fact(value: EvidenceFact | Mapping[str, Any]) -> EvidenceFact:
    if isinstance(value, EvidenceFact):
        return value
    payload = dict(value)
    return EvidenceFact(
        fact_id=str(payload.get("fact_id") or "evidence_fact"),
        fact_type=str(payload.get("fact_type") or "visible_evidence"),
        unit_tokens=tuple(
            str(item).strip() for item in _list_value(payload.get("unit_tokens")) if str(item).strip()
        ),
        provenance_refs=tuple(
            str(item).strip()
            for item in _list_value(payload.get("provenance_refs"))
            if str(item).strip()
        ),
        authority_labels=tuple(
            str(item).strip()
            for item in _list_value(payload.get("authority_labels"))
            if str(item).strip()
        ),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _optional_float(value)
        if parsed is not None:
            return parsed
    return None


def _stable_public_hash(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _public_projection_json(projection: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = _mapping_or_none(projection.get("diagnostics")) or {}
    compiler = _mapping_or_none(projection.get("compiler")) or {}
    shared_graph = _mapping_or_none(projection.get("shared_graph")) or {}
    adapter_guard = _mapping_or_none(projection.get("adapter_execution_guard")) or {}
    terminal_message_policy = _mapping_or_none(
        projection.get("terminal_completion_message_policy")
    ) or {}
    request_contract = _mapping_or_none(projection.get("request_contract")) or {}
    request_contract_coverage = (
        _mapping_or_none(projection.get("request_contract_coverage")) or {}
    )
    return {
        "decision": _public_safe_text(str(projection.get("decision") or "")),
        "request_contract_parse_error_codes": [
            _public_safe_text(str(item))
            for item in _list_value(
                projection.get("request_contract_parse_error_codes")
            )
        ],
        "projected_footprint_size": _int_value(projection.get("projected_footprint_size")),
        "trimmed_footprint_size": _int_value(projection.get("trimmed_footprint_size")),
        "readmitted_footprint_size": _int_value(
            projection.get("readmitted_footprint_size")
        )
        or len(_list_value(projection.get("readmitted_footprint"))),
        "progress_units": _int_value(projection.get("progress_units")),
        "discharged_obligation_count": _int_value(
            projection.get("discharged_obligation_count")
        ),
        "makes_progress": bool(projection.get("makes_progress")),
        "carrier_exact_match_required": bool(
            projection.get("carrier_exact_match_required")
        ),
        "selected_obligation_count": len(
            _list_value(projection.get("selected_obligation_ids"))
        ),
        "selected_obligation_id_hashes": [
            _stable_public_hash(str(item))
            for item in _list_value(projection.get("selected_obligation_ids"))
            if str(item).strip()
        ],
        "legacy_require_full_discharge_argument_observed": bool(
            projection.get("legacy_require_full_discharge_argument_observed")
        ),
        "shared_compiler_full_discharge_policy": _public_safe_text(
            str(projection.get("shared_compiler_full_discharge_policy") or "")
        ),
        "diagnostics": _public_diagnostics_json(diagnostics),
        "compiler": _public_compiler_json(compiler),
        "adapter_execution_guard": _public_adapter_execution_guard_json(adapter_guard),
        "shared_frontier": _public_shared_frontier_json(
            _mapping_or_none(projection.get("shared_frontier")) or {}
        ),
        "shared_prompt_contract": _public_shared_prompt_contract_json(
            _mapping_or_none(projection.get("shared_prompt_contract")) or {}
        ),
        "terminal_completion_message_policy": {
            "enabled": bool(terminal_message_policy.get("enabled")),
            "selected_dependent_message_count": _int_value(
                terminal_message_policy.get("selected_dependent_message_count")
            ),
            "message_nodes_do_not_increase_tool_suffix_budget": bool(
                terminal_message_policy.get(
                    "message_nodes_do_not_increase_tool_suffix_budget"
                )
            ),
        },
        "request_contract": {
            "schema": _public_safe_text(str(request_contract.get("schema") or "")),
            "surface": _public_safe_text(str(request_contract.get("surface") or "")),
            "status": _public_safe_text(str(request_contract.get("status") or "")),
            "obligation_count": _int_value(
                request_contract.get("obligation_count")
            ),
            "communication_requirement_count": _int_value(
                request_contract.get("communication_requirement_count")
            ),
            "required_terminal_literal_count": _int_value(
                request_contract.get("required_terminal_literal_count")
            ),
            "all_or_nothing": bool(request_contract.get("all_or_nothing")),
            "contract_sha256": _public_safe_text(
                str(request_contract.get("contract_sha256") or "")
            ),
        },
        "request_contract_coverage": {
            "complete": bool(request_contract_coverage.get("complete")),
            "covered_obligation_count": _int_value(
                request_contract_coverage.get("covered_obligation_count")
            ),
            "missing_obligation_count": _int_value(
                request_contract_coverage.get("missing_obligation_count")
            ),
            "extra_write_count": _int_value(
                request_contract_coverage.get("extra_write_count")
            ),
            "dependency_error_count": _int_value(
                request_contract_coverage.get("dependency_error_count")
            ),
            "missing_terminal_literal_count": _int_value(
                request_contract_coverage.get("missing_terminal_literal_count")
            ),
        },
        "shared_graph": {
            "node_count": _int_value(shared_graph.get("node_count")),
            "binding_count": _int_value(shared_graph.get("binding_count")),
            "stable_node_id_hashes": [
                _stable_public_hash(str(item))
                for item in _list_value(shared_graph.get("stable_node_ids"))
                if str(item).strip()
            ],
        },
    }


def _public_diagnostics_json(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": _public_safe_text(str(diagnostics.get("schema") or "")),
        "abstain_bucket": _public_safe_text(
            str(diagnostics.get("abstain_bucket") or "")
        ),
        "candidate_node_count": _int_value(diagnostics.get("candidate_node_count")),
        "tool_call_node_count": _int_value(diagnostics.get("tool_call_node_count")),
        "read_call_node_count": _int_value(diagnostics.get("read_call_node_count")),
        "message_node_count": _int_value(diagnostics.get("message_node_count")),
        "valid_typed_node_count_before_pi": _int_value(
            diagnostics.get("valid_typed_node_count_before_pi")
        ),
        "committed_effect_node_count": _int_value(
            diagnostics.get("committed_effect_node_count")
        ),
        "evidence_citation_reject_count": _int_value(
            diagnostics.get("evidence_citation_reject_count")
        ),
        "frontier_exact_match_node_count": _int_value(
            diagnostics.get("frontier_exact_match_node_count")
        ),
        "frontier_extra_evidence_bound_node_count": _int_value(
            diagnostics.get("frontier_extra_evidence_bound_node_count")
        ),
        "frontier_rejected_node_count": _int_value(
            diagnostics.get("frontier_rejected_node_count")
        ),
        "parse_error_available": bool(diagnostics.get("parse_error_available")),
        "parse_error_hash": (
            _stable_public_hash(str(diagnostics.get("parse_error_hash")))
            if diagnostics.get("parse_error_hash")
            else ""
        ),
        "reject_reason_counts": _public_safe_count_mapping(
            _mapping_or_none(diagnostics.get("reject_reason_counts")) or {}
        ),
    }


def _public_compiler_json(compiler: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "decision": _public_safe_text(str(compiler.get("decision") or "")),
        "selected_node_count": len(_list_value(compiler.get("selected_node_ids"))),
        "selected_node_id_hashes": [
            _stable_public_hash(str(item))
            for item in _list_value(compiler.get("selected_node_ids"))
            if str(item).strip()
        ],
        "selected_authority_label": _public_safe_text(
            str(compiler.get("selected_authority_label") or "")
        ),
        "suffix_payload_count": len(_list_value(compiler.get("suffix_payloads"))),
        "compile_errors": _public_safe_strings(
            str(item) for item in _list_value(compiler.get("compile_errors"))
        ),
        "minimum_authority_trace_count": len(
            _list_value(compiler.get("minimum_authority_trace"))
        ),
    }


def _public_adapter_execution_guard_json(guard: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "decision": _public_safe_text(str(guard.get("decision") or "")),
        "compile_errors": _public_safe_strings(
            str(item) for item in _list_value(guard.get("compile_errors"))
        ),
        "reason": _public_safe_text(str(guard.get("reason") or "")),
    }


def _public_shared_prompt_contract_json(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "compiler_setting": _public_safe_text(str(contract.get("compiler_setting") or "")),
        "mode": _public_safe_text(str(contract.get("mode") or "")),
        "authority_lattice_labels": _public_safe_strings(
            str(item) for item in _list_value(contract.get("authority_lattice_labels"))
        ),
        "allowed_authorities": _public_safe_strings(
            str(item) for item in _list_value(contract.get("allowed_authorities"))
        ),
        "strict_allowed_authorities": _public_safe_strings(
            str(item) for item in _list_value(contract.get("strict_allowed_authorities"))
        ),
        "rejected_authorities": _public_safe_strings(
            str(item) for item in _list_value(contract.get("rejected_authorities"))
        ),
        "collision_budget": _first_float(contract.get("collision_budget")),
        "suffix_tool_call_budget": _int_value(contract.get("suffix_tool_call_budget")),
        "require_action_receipt_witness": bool(
            contract.get("require_action_receipt_witness")
        ),
        "default_off": bool(contract.get("default_off")),
    }


def _public_shared_frontier_json(frontier: Mapping[str, Any]) -> dict[str, Any]:
    policy = _mapping_or_none(frontier.get("verified_carrier_policy")) or {}
    contract = _mapping_or_none(frontier.get("contract_summary")) or {}
    return {
        "frontier_timing": _public_safe_text(str(frontier.get("frontier_timing") or "")),
        "allowed_effect_unit_count": len(_list_value(frontier.get("allowed_effect_units"))),
        "evidence_fact_id_count": len(_list_value(frontier.get("evidence_fact_ids"))),
        "contract_summary": {
            "closed_obligation_count": _int_value(contract.get("closed_obligation_count")),
            "residual_obligation_count": _int_value(
                contract.get("residual_obligation_count")
            ),
            "forbidden_write_count": _int_value(contract.get("forbidden_write_count")),
            "state_change_required": bool(contract.get("state_change_required")),
        },
        "verified_carrier_policy": {
            "policy": _public_safe_text(str(policy.get("policy") or "")),
            "carrier_sets": _public_safe_strings(
                str(item) for item in _list_value(policy.get("carrier_sets"))
            ),
            "verified_carrier_ref_count": _int_value(
                policy.get("verified_carrier_ref_count")
            ),
            "pending_carrier_ref_count": _int_value(
                policy.get("pending_carrier_ref_count")
            ),
            "executed_commitment_ref_count": _int_value(
                policy.get("executed_commitment_ref_count")
            ),
        },
    }


def _public_selected_tool_names(
    nodes: Sequence[CFRCProposalNode],
    *,
    allowed_tool_names: set[str],
) -> list[str]:
    names = [
        node.tool_name
        for node in nodes
        if node.kind == "tool_call"
        and node.tool_name in allowed_tool_names
        and _is_public_safe_text(node.tool_name)
    ]
    return sorted(dict.fromkeys(names))


def _public_safe_text(value: str) -> str:
    text = str(value).strip()
    return text if _is_public_safe_text(text) else ""


def _public_safe_strings(values: Iterable[str]) -> list[str]:
    return sorted({text for value in values if (text := _public_safe_text(value))})


def _public_safe_count_mapping(values: Mapping[str, Any]) -> dict[str, int]:
    safe: dict[str, int] = {}
    for key, value in values.items():
        safe_key = _public_safe_text(str(key))
        if not safe_key:
            continue
        safe[safe_key] = _int_value(value)
    return dict(sorted(safe.items()))


def _public_safe_argument_keys(arguments: Mapping[str, Any]) -> list[str]:
    keys: list[str] = []
    for raw_key in arguments:
        key = str(raw_key).strip()
        if not _is_public_safe_text(key):
            continue
        keys.append(key)
    return sorted(set(keys))


def _is_public_safe_text(value: str) -> bool:
    text = str(value).strip()
    if not text:
        return False
    lowered = text.lower()
    token_text = lowered.replace("-", "_").replace(" ", "_")
    if text.startswith("/") or "/" in text or "\\" in text or ".json" in lowered:
        return False
    return not any(token in token_text for token in _FORBIDDEN_PUBLIC_SUMMARY_TOKENS)


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


__all__ = [
    "CFRCProposalGraph",
    "CFRCProposalNode",
    "CFRCProjectionResult",
    "CFRCResidualCertificate",
    "CFRC_STRONG_SYSTEM_PROMPT",
    "build_residual_certificate",
    "build_strong_prompt_payload",
    "compact_visible_evidence",
    "evidence_facts_from_visible_rows",
    "frontier_candidate_residual_obligations",
    "load_jsonl",
    "mutating_tool_names_by_heuristic",
    "normalize_tool_names",
    "parse_typed_proposal",
    "post_cfrc_cheap_control_blocked",
    "project_cfrc_proposal",
    "receipt_bound_commitments",
    "register_tau2_cfrc_agent",
    "strong_call_required",
    "suffix_final_message_from_receipts",
    "cfrc_runtime_ledger_from_results",
    "tool_effect_unit",
    "write_jsonl_append",
]

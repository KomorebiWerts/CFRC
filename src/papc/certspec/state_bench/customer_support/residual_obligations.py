"""Visible-only residual obligation digest for STATE-Bench customer_support."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from papc.certspec.reversibility import (
    DEFAULT_RESTORATION_COST,
    IRREVERSIBLE_RESTORATION_COST,
)
from papc.certspec.state_bench import request_cues
from papc.certspec.state_bench.adapter_protocol import (
    ContractJudgment,
    DialogueTriage,
    ResidualDomainAdapter,
    ResidualProposal,
    q_precheck_judgment_from_errors,
)
from papc.certspec.state_bench.customer_support import residual_gates
from papc.certspec.state_bench.customer_support.state_write_contract import (
    SURFACE_RESIDUAL,
    WRITE_TOOL_NAMES,
)
from papc.certspec.state_write_contract import (
    Obligation,
    ProposedTargetAction,
    StateWriteContract,
)

SCORER_FIELD_BLOCKLIST = frozenset(
    {
        "task_requirements_details",
        "state_requirements_details",
        "task_completion_pass",
        "state_requirements_met",
        "task_requirements_met",
    }
)
REQUEST_CUE_VERSION = request_cues.REQUEST_CUE_VERSION_V1_1
REQUEST_CUE_TERMS_V1 = (
    "cancel",
    "check",
    "exchange",
    "get",
    "help",
    "please",
    "process",
    "refund",
    "replace",
    "return",
    "tell",
    "warranty",
    "i need",
    "i want",
    "i'd like",
    "can you",
    "could you",
)
STATE_WRITE_REOPEN_TERMS = (
    "add",
    "adjust",
    "correct",
    "fix",
    "issue",
    "issued",
    "issuing",
    "please",
    "process",
    "put",
    "return",
)
DIALOGUE_ONLY_TERMS = (
    "clarify",
    "confirm",
    "does that include",
    "explain",
    "i understand",
    "make sure",
    "thanks",
    "what is",
)
COST_UNIT_FAMILIES = residual_gates.COST_UNIT_FAMILIES
DERIVED_UNIT_FAMILIES = frozenset({"customer_support.preview_result"})

# Every effect unit a write tool can contribute to a proposal footprint: a
# non-confirm call yields the preview unit, a confirm call adds the confirm
# unit (mirrors ``_call_units``). Consumers may drop a tool schema only when
# this whole set misses the certified scope; reads are absent and always kept.
CERTIFIED_PROJECTION_TOOL_UNITS: Mapping[str, frozenset[str]] = {
    name: frozenset(
        {
            f"customer_support.preview.{name}",
            f"customer_support.confirm.{name}",
        }
    )
    for name in sorted(WRITE_TOOL_NAMES)
}


@dataclass(frozen=True)
class CustomerSupportResidualAdapter:
    surface: str = "state_bench_customer_support"
    write_tool_names: frozenset[str] = frozenset(WRITE_TOOL_NAMES)
    cost_unit_families: frozenset[str] = COST_UNIT_FAMILIES
    derived_unit_families: frozenset[str] = DERIVED_UNIT_FAMILIES
    # Reads that ground customer-entitlement state (tier, repeat-customer
    # standing) before fee-bearing or benefit-bearing closure.
    account_read_tool_names: frozenset[str] = frozenset({"get_customer"})
    # Maximal per-tool effect-unit sets for certified tool-surface projection;
    # tools absent from this mapping are never dropped.
    certified_projection_tool_units: ClassVar[Mapping[str, frozenset[str]]] = (
        CERTIFIED_PROJECTION_TOOL_UNITS
    )

    def build_obligation_digest(
        self,
        payload: Mapping[str, Any],
        tool_schemas: Mapping[str, Any],
    ) -> dict[str, Any]:
        return build_residual_obligation_digest(payload, tool_schemas)

    def extract_dialogue_obligations(
        self, payload: Mapping[str, Any]
    ) -> DialogueTriage:
        return DialogueTriage()

    def dialogue_progress_units(
        self,
        payload: Mapping[str, Any],
        proposal: Any,
    ) -> tuple[str, ...]:
        del payload, proposal
        return ()

    def visible_evidence_tokens(self, payload: Mapping[str, Any]) -> set[str]:
        return residual_gates.visible_evidence_tokens(payload)

    def visible_conversation(self, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
        return residual_gates.visible_conversation(payload)

    def q_precheck_errors(self, payload: Mapping[str, Any], proposal: Any) -> list[str]:
        return residual_gates.preview_confirm_errors(payload, proposal)

    def q_precheck_judgment(
        self,
        payload: Mapping[str, Any],
        proposal: Any,
    ) -> ContractJudgment:
        return q_precheck_judgment_from_errors(
            self.q_precheck_errors(payload, proposal)
        )

    def deterministic_seeds(self, payload: Mapping[str, Any]) -> list[ResidualProposal]:
        return []

    def state_write_contract(self, payload: Mapping[str, Any]) -> StateWriteContract:
        committed = frozenset(
            f"customer_support.confirm.{str(call.get('name') or 'write')}#committed{index}"
            for index, call in enumerate(
                residual_gates.confirmed_write_records(payload), start=1
            )
        )
        residual_witnesses = _residual_tool_witnesses(payload)
        residual_tools = frozenset(residual_witnesses)
        residual = frozenset(
            Obligation(
                obligation_id=f"customer_support.residual.{tool_name}",
                write_footprint=frozenset(
                    {
                        f"customer_support.preview.{tool_name}",
                        f"customer_support.confirm.{tool_name}",
                    }
                ),
                postcondition_witness=residual_witnesses.get(
                    tool_name,
                    "visible_preview_or_request_scope_write_tool",
                ),
                surface=SURFACE_RESIDUAL,
                status="open_bounded",
            )
            for tool_name in sorted(residual_tools)
        )
        return StateWriteContract(
            surface=SURFACE_RESIDUAL,
            derivation_timing="runtime_progressive",
            closed=frozenset(
                Obligation(
                    obligation_id=f"customer_support.committed.{unit}",
                    write_footprint=frozenset({unit}),
                    postcondition_witness="visible_confirmed_customer_support_write",
                    surface=SURFACE_RESIDUAL,
                    status="closed",
                )
                for unit in sorted(committed)
            ),
            residual=residual,
            forbidden_writes=committed,
            legal_actions=frozenset(
                {"bounded_repair", "abstain"} if residual else {"abstain"}
            ),
            state_change_required=bool(residual),
            commitment_call_count=len(committed),
            commitment_footprint_size=len(committed),
            commitment_app_count=1 if committed else 0,
            soundness_assumptions=(
                "customer_support_units_are_preview_confirm_families_not_item_ids",
                "confirm_units_are_schema_irreversible_lifecycle_writes",
            ),
            touched_units=committed
            | frozenset(
                unit for obligation in residual for unit in obligation.write_footprint
            ),
        )

    def to_target_action(
        self,
        proposal: Any,
        payload: Mapping[str, Any],
    ) -> ProposedTargetAction:
        del payload
        groups = tuple(
            group
            for group in (_call_units(call) for call in _proposal_tool_calls(proposal))
            if group
        )
        write_footprint = frozenset(unit for group in groups for unit in group)
        dependencies = tuple(
            (
                f"customer_support.confirm.{tool_name}",
                frozenset({f"customer_support.preview.{tool_name}"}),
            )
            for tool_name in sorted(_confirm_tool_names(proposal))
        )
        return ProposedTargetAction(
            surface=SURFACE_RESIDUAL,
            action_id=f"customer_support_residual:{_proposal_action(proposal)}",
            write_footprint=write_footprint,
            terminal_completion=bool(
                _proposal_action(proposal) == "execute" and not write_footprint
            ),
            effect_groups=groups,
            dependencies=dependencies,
        )

    def restoration_cost_vector(
        self,
        payload: Mapping[str, Any],
        contract: StateWriteContract,
        proposed_action: ProposedTargetAction,
    ) -> Mapping[str, float] | None:
        del payload
        units = contract.forbidden_writes | proposed_action.write_footprint
        return {unit: _restoration_cost_for_unit(unit) for unit in sorted(units)}


ADAPTER: ResidualDomainAdapter = CustomerSupportResidualAdapter()


def _residual_tool_names(payload: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(_residual_tool_witnesses(payload))


def _residual_tool_witnesses(payload: Mapping[str, Any]) -> dict[str, str]:
    pending_preview_tools = {
        str(call.get("name") or "")
        for call in _pending_preview_records(payload)
        if str(call.get("name") or "") in WRITE_TOOL_NAMES
    }
    request_quote = _first_user_request_quote(payload)
    witnesses = {
        tool_name: request_quote or f"visible_preview_record:{tool_name}"
        for tool_name in sorted(pending_preview_tools)
    }
    for tool_name, witness in _user_state_write_tool_witnesses(payload).items():
        if _tool_family_still_pending(payload, tool_name):
            witnesses.setdefault(tool_name, witness)
    return {
        name: witness
        for name, witness in sorted(witnesses.items())
        if name in WRITE_TOOL_NAMES
    }


def _pending_preview_records(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for event in _conversation_events(payload):
        if not event["is_preview"]:
            continue
        tool_name = str(event["tool_name"])
        latest_confirm = _latest_confirm_event_index(payload, tool_name)
        if latest_confirm is None or int(event["event_index"]) > latest_confirm:
            rows.append(event["call"])
    return rows


def _user_state_write_tool_witnesses(payload: Mapping[str, Any]) -> dict[str, str]:
    witnesses: dict[str, str] = {}
    for message in _conversation(payload):
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "").lower()
        if _cancels_pending_customer_support_write(content):
            witnesses.clear()
            continue
        quote = _first_matching_sentence(str(message.get("content") or ""))
        for tool_name in sorted(WRITE_TOOL_NAMES):
            if _is_state_write_reopen_request(tool_name, content):
                witnesses.setdefault(
                    tool_name,
                    quote or f"visible_user_{tool_name}_request",
                )
    return witnesses


def _has_unnegated_cancel_clause(text: str) -> bool:
    if not _contains_any_term(text, ("cancel",)):
        return False
    negated_patterns = (
        r"\bdo\s+not\s+cancel\b",
        r"\bdon't\s+cancel\b",
        r"\bdoes\s+not\s+cancel\b",
        r"\bshould\s+not\s+cancel\b",
        r"\bmust\s+not\s+cancel\b",
        r"\bnot\s+cancel\b",
    )
    return not any(re.search(pattern, text) for pattern in negated_patterns)


def _tool_family_still_pending(payload: Mapping[str, Any], tool_name: str) -> bool:
    latest_confirm = _latest_confirm_event_index(payload, tool_name)
    if latest_confirm is None:
        return True
    if _has_pending_preview_after(payload, tool_name, latest_confirm):
        return True
    return _has_user_write_request_after(payload, tool_name, latest_confirm)


def _has_pending_preview_after(
    payload: Mapping[str, Any],
    tool_name: str,
    latest_confirm: int,
) -> bool:
    return any(
        event["tool_name"] == tool_name
        and event["is_preview"] is True
        and int(event["event_index"]) > latest_confirm
        for event in _conversation_events(payload)
    )


def _has_user_write_request_after(
    payload: Mapping[str, Any],
    tool_name: str,
    latest_confirm: int,
) -> bool:
    terms = _tool_family_terms(tool_name)
    for event in _conversation_events(payload):
        if event["role"] != "user" or int(event["event_index"]) <= latest_confirm:
            continue
        text = str(event["content"]).lower()
        if not any(term in text for term in terms):
            continue
        if any(term in text for term in DIALOGUE_ONLY_TERMS) and not _contains_any_term(
            text,
            ("add", "adjust", "correct", "fix", "issue", "issued", "issuing", "put"),
        ):
            continue
        if _is_state_write_reopen_request(tool_name, text):
            return True
    return False


def _is_state_write_reopen_request(tool_name: str, text: str) -> bool:
    if tool_name == "process_refund":
        return any(
            phrase in text
            for phrase in (
                "please refund",
                "process refund",
                "process the refund",
                "issue refund",
                "issue a refund",
                "issuing the refund",
                "refund me",
                "want a refund",
                "need a refund",
                "like a refund",
                "missing credit",
                "issue credit",
                "issue the credit",
                "issuing the credit",
                "put the credit",
            )
        )
    if tool_name == "process_return":
        return any(
            phrase in text
            for phrase in (
                "please return",
                "process return",
                "process the return",
                "return item",
                "return both",
                "start a return",
                "want to return",
                "like to return",
            )
        )
    if tool_name == "process_exchange":
        return any(
            phrase in text
            for phrase in (
                "please exchange",
                "process exchange",
                "process the exchange",
                "start exchange",
                "start the exchange",
                "want to exchange",
                "like to exchange",
            )
        )
    if tool_name == "process_warranty_claim":
        return _contains_any_term(text, ("warranty", "replace", "replacement")) and (
            _contains_any_term(text, ("please", "process", "start"))
        )
    if tool_name == "cancel_order":
        return _has_unnegated_cancel_clause(text) and _contains_any_term(
            text, ("please", "process")
        )
    return _contains_any_term(text, STATE_WRITE_REOPEN_TERMS)


def _cancels_pending_customer_support_write(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "leave the order unchanged",
            "leave ord-",
            "leave it unchanged",
            "leave everything unchanged",
            "not taking any action",
            "no action yet",
            "that's all i needed",
            "that is all i needed",
        )
    )


def _contains_any_term(text: str, terms: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms)


def _tool_family_terms(tool_name: str) -> tuple[str, ...]:
    if tool_name == "cancel_order":
        return ("cancel",)
    if tool_name == "process_exchange":
        return ("exchange",)
    if tool_name == "process_warranty_claim":
        return ("warranty", "replace", "replacement", "defective")
    if tool_name == "process_return":
        return ("return",)
    if tool_name == "process_refund":
        return ("refund", "credit")
    return (tool_name,)


def _latest_confirm_event_index(
    payload: Mapping[str, Any], tool_name: str
) -> int | None:
    indexes = [
        int(event["event_index"])
        for event in _conversation_events(payload)
        if event["tool_name"] == tool_name and event["is_confirm"] is True
    ]
    return max(indexes) if indexes else None


def _conversation_events(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    event_index = 0
    for message in _conversation(payload):
        role = str(message.get("role") or "")
        events.append(
            {
                "event_index": event_index,
                "role": role,
                "content": str(message.get("content") or ""),
                "tool_name": "",
                "is_preview": False,
                "is_confirm": False,
                "call": {},
            }
        )
        event_index += 1
        raw_calls = message.get("tool_calls")
        if not isinstance(raw_calls, list):
            continue
        for call in raw_calls:
            if not isinstance(call, Mapping):
                continue
            events.append(
                {
                    "event_index": event_index,
                    "role": "tool",
                    "content": "",
                    "tool_name": str(call.get("name") or ""),
                    "is_preview": residual_gates.is_preview_call(call),
                    "is_confirm": residual_gates.is_confirm_call(call),
                    "call": call,
                }
            )
            event_index += 1
    return events


def _proposal_tool_calls(proposal: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(proposal, Mapping) and isinstance(proposal.get("tool_calls"), list):
        return tuple(
            call for call in proposal["tool_calls"] if isinstance(call, Mapping)
        )
    calls = getattr(proposal, "tool_calls", ())
    if isinstance(calls, tuple):
        return calls
    if isinstance(calls, list):
        return tuple(call for call in calls if isinstance(call, Mapping))
    return ()


def _call_units(call: Mapping[str, Any]) -> frozenset[str]:
    name = str(call.get("name") or "")
    if name not in WRITE_TOOL_NAMES:
        return frozenset()
    preview = f"customer_support.preview.{name}"
    confirm = f"customer_support.confirm.{name}"
    if residual_gates.is_confirm_call(call):
        return frozenset({preview, confirm})
    return frozenset({preview})


def _confirm_tool_names(proposal: Any) -> frozenset[str]:
    return frozenset(
        str(call.get("name") or "")
        for call in _proposal_tool_calls(proposal)
        if residual_gates.is_confirm_call(call)
    )


def _restoration_cost_for_unit(unit: str) -> float:
    if ".confirm." in unit:
        return IRREVERSIBLE_RESTORATION_COST
    return DEFAULT_RESTORATION_COST


def _proposal_action(proposal: Any) -> str:
    if isinstance(proposal, Mapping):
        return str(proposal.get("action") or "")
    return str(getattr(proposal, "action", "") or "")


def _first_user_request_quote(payload: Mapping[str, Any]) -> str:
    for message in _conversation(payload):
        if message.get("role") != "user":
            continue
        quote = _first_matching_sentence(str(message.get("content") or ""))
        if quote:
            return quote
    return ""


def _first_matching_sentence(content: str) -> str:
    for sentence in _sentences(content):
        if sentence.endswith("?") or _begins_with_request_cue(sentence):
            return sentence
    return ""


def build_residual_obligation_digest(
    payload: Mapping[str, Any],
    tool_schemas: Mapping[str, Any],
) -> dict[str, Any]:
    conversation = _conversation(payload)
    return {
        "request_cue_version": REQUEST_CUE_VERSION,
        "user_requests": _user_requests(conversation),
        "reached_state_snapshot": _reached_state_snapshot(conversation),
        "committed_writes": _committed_writes(conversation, tool_schemas),
        "binding_slots": _binding_slots(payload),
        "visible_tool_errors": _visible_tool_errors(conversation),
    }


def preview_confirm_target_action(
    tool_name: str,
    *,
    cost_units: frozenset[str] = frozenset(),
) -> ProposedTargetAction:
    preview_unit = f"customer_support.preview.{tool_name}"
    confirm_unit = f"customer_support.confirm.{tool_name}"
    write_footprint = frozenset({preview_unit, confirm_unit}) | cost_units
    return ProposedTargetAction(
        surface="state_bench_customer_support",
        action_id=f"preview_then_confirm:{tool_name}",
        write_footprint=write_footprint,
        effect_groups=(write_footprint,),
        dependencies=((confirm_unit, frozenset({preview_unit})),),
    )


def quote_in_visible_conversation(payload: Mapping[str, Any], quote: str) -> bool:
    if not quote.strip():
        return False
    return normalize_whitespace(quote) in normalize_whitespace(
        " ".join(
            str(message.get("content") or "")
            for message in _conversation(payload)
            if isinstance(message, Mapping)
        )
    )


def normalize_whitespace(value: str) -> str:
    return " ".join(str(value).split())


def _user_requests(conversation: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for turn_index, message in enumerate(conversation):
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "")
        for sentence in _sentences(content):
            if sentence.endswith("?") or _begins_with_request_cue(sentence):
                rows.append({"turn_index": turn_index, "quote": sentence})
    return rows


def _sentences(content: str) -> list[str]:
    return request_cues.sentences(content)


def _begins_with_request_cue(sentence: str) -> bool:
    return request_cues.begins_with_request_cue(sentence, REQUEST_CUE_TERMS_V1)


def _reached_state_snapshot(conversation: list[Mapping[str, Any]]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "orders_last_visible": [],
        "customers_last_visible": [],
        "warranty_status_last_visible": [],
    }
    latest: dict[str, Any] = {
        "get_order": [],
        "get_customer": [],
        "get_warranty_status": [],
    }
    for call in _tool_calls(conversation):
        result = call.get("result")
        if not isinstance(result, Mapping):
            continue
        name = str(call.get("name") or "")
        if name in latest:
            latest[name].append(dict(result))
    snapshot["orders_last_visible"] = latest["get_order"]
    snapshot["customers_last_visible"] = latest["get_customer"]
    snapshot["warranty_status_last_visible"] = latest["get_warranty_status"]
    return snapshot


def _committed_writes(
    conversation: list[Mapping[str, Any]],
    tool_schemas: Mapping[str, Any],
) -> list[str]:
    write_names = {str(name) for name in tool_schemas} or set(WRITE_TOOL_NAMES)
    writes: list[str] = []
    for call in _tool_calls(conversation):
        name = str(call.get("name") or "")
        if name not in write_names or not residual_gates.is_confirm_call(call):
            continue
        writes.append(f"{name}({{...}})")
    return writes


def _binding_slots(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    text = " ".join(
        str(message.get("content") or "").lower()
        for message in _conversation(payload)
        if message.get("role") == "user"
    )
    slots: dict[str, str] = {}
    if any(
        term in text for term in ("return", "refund", "exchange", "warranty", "cancel")
    ):
        if any(
            token.startswith("item")
            for token in residual_gates.visible_evidence_tokens(payload)
        ):
            slots["item_id"] = "visible_order_data"
        if "return" in text or "refund" in text:
            slots["return_reason"] = "visible_user_request"
    for preview in residual_gates.preview_records(payload):
        result = (
            preview.get("result") if isinstance(preview.get("result"), Mapping) else {}
        )
        if "refund_amount" in result:
            slots["refund_amount"] = "preview_call"
        if "restocking_fee" in result:
            slots["restocking_fee"] = "preview_call"
        if "refund_method" in result:
            slots["refund_method"] = "preview_call"
        if "goodwill_credit" in result:
            slots["goodwill_credit"] = "preview_call"
        if "goodwill_credit_method" in result:
            slots["goodwill_credit_method"] = "preview_call"
    return [
        {"slot_name": name, "source_kind": source}
        for name, source in sorted(slots.items())
    ]


def _visible_tool_errors(conversation: list[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    for call in _tool_calls(conversation):
        result = call.get("result")
        if not isinstance(result, Mapping):
            continue
        status = str(result.get("status") or "").lower()
        if "error" in result or status in {"error", "failed", "failure"}:
            errors.append(str(call.get("name") or "tool"))
    return errors


def _tool_calls(conversation: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    calls: list[Mapping[str, Any]] = []
    for message in conversation:
        raw_calls = message.get("tool_calls")
        if not isinstance(raw_calls, list):
            continue
        calls.extend(call for call in raw_calls if isinstance(call, Mapping))
    return calls


def _conversation(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = payload.get("conversation")
    if not isinstance(raw, list):
        return []
    return [message for message in raw if isinstance(message, Mapping)]

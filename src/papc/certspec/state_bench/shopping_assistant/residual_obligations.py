"""Visible-only residual obligation digest for STATE-Bench Shopping."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from papc.certspec.state_bench import request_cues
from papc.certspec.state_bench.adapter_protocol import (
    ContractJudgment,
    DialogueTriage,
    ResidualDomainAdapter,
    ResidualProposal,
    q_precheck_judgment_from_errors,
)
from papc.certspec.state_bench.shopping_assistant import (
    residual_gates,
    state_write_contract,
)
from papc.certspec.state_write_contract import ProposedTargetAction, StateWriteContract

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
HINT_EXTRACTOR_VERSION = residual_gates.HINT_EXTRACTOR_VERSION

REQUEST_CUE_TERMS = (
    "add",
    "remove",
    "find",
    "show",
    "compare",
    "check",
    "tell",
    "recommend",
    "explain",
    "describe",
    "confirm",
    "quote",
    "list",
    "update",
    "set",
    "change",
    "apply",
    "ship",
    "order",
    "buy",
    "get",
    "include",
    "can you",
    "could you",
    "please",
    "i want",
    "i need",
    "i'd like",
    "let's",
    "lets",
)
REQUEST_CUE_TERMS_V1 = REQUEST_CUE_TERMS
DISCOURSE_MARKERS_V1_1 = request_cues.DISCOURSE_MARKERS_V1_1
WRITE_TOOL_NAMES = state_write_contract.WRITE_TOOL_NAMES
CART_ITEM_WRITE_TOOL_NAMES = frozenset(
    {
        "add_to_cart",
        "update_cart_item",
        "remove_from_cart",
    }
)
COST_UNIT_FAMILIES = frozenset(
    {
        "carts.subtotal",
        "carts.total",
        "carts.promotions",
        "carts.loyalty_points",
        "carts.shipping_option",
    }
)
DERIVED_UNIT_FAMILIES = frozenset(
    {
        "dialogue.shopping.policy_quote",
        "dialogue.shopping.comparison",
        "dialogue.shopping.confirmation",
        "dialogue.shopping.dialogue",
    }
)


@dataclass(frozen=True)
class ShoppingResidualAdapter:
    surface: str = "state_bench_shopping_assistant"
    write_tool_names: frozenset[str] = WRITE_TOOL_NAMES
    cost_unit_families: frozenset[str] = COST_UNIT_FAMILIES
    derived_unit_families: frozenset[str] = DERIVED_UNIT_FAMILIES
    # Reads that ground customer-entitlement state (tier, first-time status,
    # loyalty balance) before pricing-bearing closure.
    account_read_tool_names: frozenset[str] = frozenset({"get_customer_account"})
    # Maximal per-tool effect-unit sets for certified tool-surface projection;
    # tools absent from this mapping are never dropped.
    certified_projection_tool_units: ClassVar[Mapping[str, frozenset[str]]] = (
        state_write_contract.CERTIFIED_PROJECTION_TOOL_UNITS
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
        open_units = tuple(
            f"dialogue.shopping.{family}"
            for family in residual_gates.open_dialogue_obligation_families(payload)
        )
        return DialogueTriage(open=open_units)

    def selector_protocol_dialogue_units(
        self,
        payload: Mapping[str, Any],
    ) -> frozenset[str]:
        return residual_gates.selector_protocol_dialogue_units(payload)

    def selector_frontier_position(
        self,
        payload: Mapping[str, Any],
        *,
        open_dialogue_units: tuple[str, ...],
        candidate_class: str,
    ) -> tuple[int, int | None] | None:
        return residual_gates.selector_frontier_position(
            payload,
            open_dialogue_units=open_dialogue_units,
            candidate_class=candidate_class,
        )

    def dialogue_progress_units(
        self,
        payload: Mapping[str, Any],
        proposal: Any,
    ) -> tuple[str, ...]:
        if residual_gates.proposal_tool_calls(proposal):
            return ()
        assistant_message = _proposal_assistant_message(proposal)
        families = residual_gates.open_dialogue_obligation_families(payload)
        if residual_gates.message_only_dialogue_errors(payload, assistant_message):
            return ()
        return tuple(f"dialogue.shopping.{family}" for family in families)

    def visible_evidence_tokens(self, payload: Mapping[str, Any]) -> set[str]:
        return residual_gates.visible_evidence_tokens(payload)

    def visible_conversation(self, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
        return residual_gates.visible_conversation(payload)

    def q_precheck_errors(self, payload: Mapping[str, Any], proposal: Any) -> list[str]:
        assistant_message = _proposal_assistant_message(proposal)
        tool_calls = residual_gates.proposal_tool_calls(proposal)
        if tool_calls:
            return []
        return residual_gates.message_only_dialogue_errors(payload, assistant_message)

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
        return state_write_contract.state_write_contract_from_payload(payload)

    def repair_targets(self, payload: Mapping[str, Any]) -> frozenset[str]:
        """Committed reversible families visibly requested to change (ell2 trigger)."""

        return state_write_contract.committed_change_repair_targets(payload)

    def to_target_action(
        self,
        proposal: Any,
        payload: Mapping[str, Any],
    ) -> ProposedTargetAction:
        return state_write_contract.proposed_target_action_from_proposal(
            proposal,
            payload,
        )

    def restoration_cost_vector(
        self,
        payload: Mapping[str, Any],
        contract: StateWriteContract,
        proposed_action: ProposedTargetAction,
    ) -> Mapping[str, float] | None:
        del payload
        units = contract.forbidden_writes | proposed_action.write_footprint
        return state_write_contract.restoration_cost_vector(sorted(units))


ADAPTER: ResidualDomainAdapter = ShoppingResidualAdapter()


def build_residual_obligation_digest(
    payload: Mapping[str, Any],
    tool_schemas: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a deterministic digest from visible conversation data only."""

    conversation = _conversation(payload)
    return {
        "request_cue_version": REQUEST_CUE_VERSION,
        "hint_extractor_version": HINT_EXTRACTOR_VERSION,
        "user_requests": _user_requests(conversation),
        "visible_obligation_hints": visible_obligation_hints(payload),
        "reached_state_snapshot": _reached_state_snapshot(conversation),
        "committed_writes": _committed_writes(conversation, tool_schemas),
        "visible_tool_errors": _visible_tool_errors(conversation),
    }


def visible_obligation_hints(payload: Mapping[str, Any]) -> dict[str, list[str]]:
    """Visible-only recall hints; these are not scorer-derived obligations."""

    dialogue = [
        f"dialogue.shopping.{family}"
        for family in residual_gates.dialogue_obligation_hint_families(payload)
    ]
    numeric = [
        f"final_summary.shopping.{family}"
        for family in sorted(
            residual_gates.final_summary_required_numeric_tokens(payload)
        )
    ]
    return {
        "dialogue_families": dialogue,
        "final_summary_numeric_families": numeric,
    }


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


def _proposal_assistant_message(proposal: Any) -> str:
    if isinstance(proposal, Mapping):
        return str(proposal.get("assistant_message") or "")
    return str(getattr(proposal, "assistant_message", "") or "")


def _conversation(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = payload.get("conversation")
    if not isinstance(raw, list):
        return []
    return [message for message in raw if isinstance(message, Mapping)]


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
    return request_cues.begins_with_request_cue(sentence, REQUEST_CUE_TERMS)


def _cue_candidate_texts(sentence: str) -> list[str]:
    return request_cues.cue_candidate_texts(sentence)


def _strip_leading_discourse_markers(sentence: str) -> str:
    return request_cues.strip_leading_discourse_markers(sentence)


def _reached_state_snapshot(conversation: list[Mapping[str, Any]]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "cart_items_last_visible": [],
        "shipping_option_last_visible": "",
        "subtotal_last_visible": None,
        "total_last_visible": None,
    }
    for call in _tool_calls(conversation):
        tool_name = str(call.get("name") or "")
        result = call.get("result")
        if not isinstance(result, Mapping):
            continue
        if isinstance(result.get("items"), list):
            snapshot["cart_items_last_visible"] = [
                dict(item) if isinstance(item, Mapping) else item
                for item in result["items"]
            ]
        else:
            _apply_single_item_result(snapshot, result, tool_name=tool_name)
        if "shipping_option" in result:
            snapshot["shipping_option_last_visible"] = (
                result.get("shipping_option") or ""
            )
        if "subtotal" in result:
            snapshot["subtotal_last_visible"] = result.get("subtotal")
        elif "cart_subtotal" in result:
            snapshot["subtotal_last_visible"] = result.get("cart_subtotal")
        if "total" in result:
            snapshot["total_last_visible"] = result.get("total")
        elif "cart_total" in result:
            snapshot["total_last_visible"] = result.get("cart_total")
    return snapshot


def _apply_single_item_result(
    snapshot: dict[str, Any],
    result: Mapping[str, Any],
    *,
    tool_name: str,
) -> None:
    item = _single_cart_item(result, tool_name=tool_name)
    if not item:
        return
    items = [
        dict(existing) if isinstance(existing, Mapping) else existing
        for existing in snapshot["cart_items_last_visible"]
    ]
    key = item.get("cart_item_id") or item.get("product_id")
    if not key:
        return
    if str(result.get("status") or "").lower() in {"removed", "deleted"}:
        snapshot["cart_items_last_visible"] = [
            existing
            for existing in items
            if not (
                isinstance(existing, Mapping)
                and (existing.get("cart_item_id") or existing.get("product_id")) == key
            )
        ]
        return
    for index, existing in enumerate(items):
        if (
            isinstance(existing, Mapping)
            and (existing.get("cart_item_id") or existing.get("product_id")) == key
        ):
            items[index] = {**existing, **item}
            snapshot["cart_items_last_visible"] = items
            return
    items.append(item)
    snapshot["cart_items_last_visible"] = items


def _single_cart_item(result: Mapping[str, Any], *, tool_name: str) -> dict[str, Any]:
    if "cart_item_id" not in result and "product_id" not in result:
        return {}
    status = str(result.get("status") or "").lower()
    if not (
        tool_name in CART_ITEM_WRITE_TOOL_NAMES
        or "cart_item_id" in result
        or status in {"added", "updated", "removed", "deleted"}
    ):
        return {}
    item_fields = (
        "cart_item_id",
        "product_id",
        "product_name",
        "quantity",
        "gift_wrap",
        "unit_price",
        "line_total",
        "variant_id",
    )
    return {
        field: result[field]
        for field in item_fields
        if field in result and result[field] is not None
    }


def _committed_writes(
    conversation: list[Mapping[str, Any]],
    tool_schemas: Mapping[str, Any],
) -> list[str]:
    write_names = {str(name) for name in tool_schemas}
    writes: list[str] = []
    for call in _tool_calls(conversation):
        name = str(call.get("name") or "")
        if name and name in write_names:
            writes.append(f"{name}({{...}})")
    return writes


def _visible_tool_errors(conversation: list[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    for call in _tool_calls(conversation):
        result = call.get("result")
        if not isinstance(result, Mapping):
            continue
        status = str(result.get("status") or "").lower()
        if "error" in result or status in {"error", "failed", "failure"}:
            name = str(call.get("name") or "tool")
            errors.append(name)
    return errors


def _tool_calls(conversation: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    calls: list[Mapping[str, Any]] = []
    for message in conversation:
        raw_calls = message.get("tool_calls")
        if not isinstance(raw_calls, list):
            continue
        calls.extend(call for call in raw_calls if isinstance(call, Mapping))
    return calls

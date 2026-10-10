"""Visible-data customer_support residual gates for STATE-Bench pilots."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from papc.certspec.state_bench.customer_support.state_write_contract import (
    TERMINAL_ITEM_STATES,
    WRITE_TOOL_NAMES,
)

ID_LIKE_RE = re.compile(
    r"\b(?:ORD|ITEM|PROD|WRT)-[A-Za-z0-9_]+\b"
    r"|\bCUST[-_][A-Za-z0-9_]+\b"
    r"|\bcust[_-][A-Za-z0-9_]+\b",
    flags=re.IGNORECASE,
)
NUMERIC_LIKE_RE = re.compile(
    r"(?<![A-Za-z0-9_-])\$?\d+(?:\.\d{1,2})?%?(?![A-Za-z0-9_-])"
)

PREVIEW_CONFIRM_TOOL_NAMES = frozenset(WRITE_TOOL_NAMES)
LIFECYCLE_WRITE_TOOL_NAMES = frozenset(
    {"process_return", "process_exchange", "process_warranty_claim"}
)
MONETARY_ONLY_WRITE_TOOL_NAMES = frozenset({"process_refund"})
SIGNED_DELIVERY_PACKAGE_NOT_FOUND_ERROR = (
    "q_precheck_fail_signed_delivery_package_not_found_item"
)
LIFECYCLE_REQUEST_CUE_TERMS = (
    "exchange",
    "return",
    "replace",
    "replacement",
    "warranty",
    "defective",
    "damaged",
    "wrong item",
    "out of stock",
)
COST_UNIT_FAMILIES = frozenset(
    {
        "order_items.refund_amount",
        "order_items.restocking_fee",
        "order_items.goodwill_credit",
        "order_items.goodwill_credit_method",
    }
)
CONFIRM_SLOT_ARG_ALIASES = {
    "refund_amount": ("amount", "refund_amount"),
    "restocking_fee": ("restocking_fee",),
    "refund_method": ("refund_method",),
    "goodwill_credit": ("goodwill_credit", "credit_amount"),
    "goodwill_credit_method": ("goodwill_credit_method", "credit_method"),
}


def extract_tool_calls_with_results(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for message in _conversation(payload):
        raw_calls = message.get("tool_calls")
        if not isinstance(raw_calls, list):
            continue
        for call in raw_calls:
            if not isinstance(call, Mapping):
                continue
            args = (
                call.get("arguments")
                if isinstance(call.get("arguments"), Mapping)
                else {}
            )
            calls.append(
                {
                    "name": str(call.get("name") or ""),
                    "arguments": dict(args),
                    "result": call.get("result"),
                }
            )
    return calls


def visible_conversation(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    visible: list[dict[str, Any]] = []
    for message in _conversation(payload):
        entry: dict[str, Any] = {
            "role": message.get("role"),
            "content": message.get("content"),
        }
        raw_calls = message.get("tool_calls")
        if isinstance(raw_calls, list):
            entry["tool_calls"] = [
                {
                    "name": call.get("name"),
                    "arguments": call.get("arguments"),
                    "result": call.get("result"),
                }
                for call in raw_calls
                if isinstance(call, Mapping)
            ]
        visible.append(entry)
    return visible


def visible_evidence_tokens(payload: Mapping[str, Any]) -> set[str]:
    blob = json.dumps(visible_conversation(payload), ensure_ascii=False)
    return {token.lower() for token in ID_LIKE_RE.findall(blob)}


def id_like_tokens(value: Any) -> set[str]:
    return {
        token for token in ID_LIKE_RE.findall(json.dumps(value, ensure_ascii=False))
    }


def visible_numeric_tokens(payload: Mapping[str, Any]) -> set[str]:
    blob = json.dumps(visible_conversation(payload), ensure_ascii=False)
    return numeric_like_tokens(blob)


def numeric_like_tokens(value: Any) -> set[str]:
    tokens: set[str] = set()
    for token in NUMERIC_LIKE_RE.findall(json.dumps(value, ensure_ascii=False)):
        tokens.update(_numeric_token_variants(token))
    return tokens


def preview_records(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for call in extract_tool_calls_with_results(payload):
        if not is_preview_call(call):
            continue
        rows.append(call)
    return rows


def confirmed_write_records(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for call in extract_tool_calls_with_results(payload):
        if not is_confirm_call(call):
            continue
        rows.append(call)
    return rows


def validate_confirm_binding(
    payload: Mapping[str, Any],
    confirm_call: Mapping[str, Any],
    *,
    additional_preview_records: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Validate that a confirm write is bound to a visible preview result."""

    if not is_confirm_call(confirm_call):
        return {"status": "not_confirm_write", "errors": [], "confirmed_slots": []}
    preview, preview_error = _matching_preview_verdict(
        payload,
        confirm_call,
        additional_preview_records=additional_preview_records,
    )
    if preview is None:
        if warranty_claim_confirm_is_visible_scope_bound(payload, confirm_call):
            return {
                "status": "trace_confirmed",
                "errors": [],
                "confirmed_slots": ["active_warranty_scope"],
                "deferred_preview_slots": [],
            }
        return {
            "status": "binding_underfill",
            "errors": [preview_error or "confirm_without_preview"],
            "confirmed_slots": [],
        }
    resolved = _resolve_deferred_preview_args(confirm_call, preview)
    if resolved["errors"]:
        return {
            "status": "binding_underfill",
            "errors": sorted(set(str(error) for error in resolved["errors"])),
            "confirmed_slots": [],
            "deferred_preview_slots": sorted(
                set(str(slot) for slot in resolved["resolved_slots"])
            ),
        }
    errors: list[str] = []
    confirmed_slots: list[str] = []
    preview_result = (
        preview.get("result") if isinstance(preview.get("result"), Mapping) else {}
    )
    confirm_args = (
        resolved["resolved_call"].get("arguments")
        if isinstance(resolved["resolved_call"].get("arguments"), Mapping)
        else {}
    )
    for slot, aliases in CONFIRM_SLOT_ARG_ALIASES.items():
        present = [name for name in aliases if name in confirm_args]
        if not present:
            continue
        confirm_value = confirm_args[present[0]]
        if confirm_value == preview_result.get(slot):
            confirmed_slots.append(slot)
        else:
            errors.append(f"{slot}_not_trace_confirmed")
    return {
        "status": "binding_underfill" if errors else "trace_confirmed",
        "errors": sorted(set(errors)),
        "confirmed_slots": sorted(set(confirmed_slots)),
        "deferred_preview_slots": sorted(
            set(str(slot) for slot in resolved["resolved_slots"])
        ),
    }


def resolve_deferred_preview_bindings(
    payload: Mapping[str, Any],
    confirm_call: Mapping[str, Any],
    *,
    additional_preview_records: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Resolve {"$from_preview": slot} confirm args from the matching preview."""

    if not is_confirm_call(confirm_call):
        return {
            "status": "not_confirm_write",
            "errors": [],
            "resolved_call": _copy_call(confirm_call),
            "resolved_slots": [],
        }
    preview, preview_error = _matching_preview_verdict(
        payload,
        confirm_call,
        additional_preview_records=additional_preview_records,
    )
    if preview is None:
        if warranty_claim_confirm_is_visible_scope_bound(payload, confirm_call):
            return {
                "status": "visible_scope_bound_without_preview",
                "errors": [],
                "resolved_call": _copy_call(confirm_call),
                "resolved_slots": [],
            }
        return {
            "status": "binding_underfill",
            "errors": [preview_error or "confirm_without_preview"],
            "resolved_call": _copy_call(confirm_call),
            "resolved_slots": [],
        }
    resolved = _resolve_deferred_preview_args(confirm_call, preview)
    errors = sorted(set(str(error) for error in resolved["errors"]))
    return {
        "status": "binding_underfill" if errors else "deferred_bindings_resolved",
        "errors": errors,
        "resolved_call": resolved["resolved_call"],
        "resolved_slots": sorted(set(str(slot) for slot in resolved["resolved_slots"])),
    }


def preview_confirm_errors(payload: Mapping[str, Any], proposal: Any) -> list[str]:
    errors: list[str] = []
    for call in proposal_tool_calls(proposal):
        verdict = validate_confirm_binding(payload, call)
        errors.extend(str(error) for error in verdict.get("errors", []))
    errors.extend(tool_family_selection_errors(payload, proposal))
    errors.extend(signed_delivery_missing_claim_errors(payload, proposal))
    return sorted(set(errors))


def warranty_claim_confirm_is_visible_scope_bound(
    payload: Mapping[str, Any],
    confirm_call: Mapping[str, Any],
) -> bool:
    """Allow non-monetary warranty claim confirms from visible active warranty evidence."""

    name = str(confirm_call.get("name") or "")
    args = (
        confirm_call.get("arguments")
        if isinstance(confirm_call.get("arguments"), Mapping)
        else {}
    )
    if name != "process_warranty_claim" or args.get("confirm") is not True:
        return False
    if any(
        alias in args
        for aliases in CONFIRM_SLOT_ARG_ALIASES.values()
        for alias in aliases
    ):
        return False
    item_id = str(args.get("item_id") or "")
    warranty_id = str(args.get("warranty_id") or "")
    if not item_id or not warranty_id:
        return False
    if not _latest_user_explicitly_requests_warranty_claim(payload):
        return False
    return any(
        _active_warranty_record_matches(call, item_id=item_id, warranty_id=warranty_id)
        for call in extract_tool_calls_with_results(payload)
        if str(call.get("name") or "") == "get_warranty_status"
    )


def tool_family_guidance(payload: Mapping[str, Any]) -> dict[str, Any]:
    cues = lifecycle_request_cues(payload)
    terminal_visible = visible_terminal_item_state(payload)
    return {
        "active_item_lifecycle_request_detected": bool(cues),
        "lifecycle_request_cues": sorted(cues),
        "visible_terminal_item_state": terminal_visible,
        "lifecycle_write_tools": sorted(LIFECYCLE_WRITE_TOOL_NAMES),
        "monetary_only_write_tools": sorted(MONETARY_ONLY_WRITE_TOOL_NAMES),
        "rule": (
            "process_refund is a monetary adjustment tool, not a substitute for "
            "closing an active item return, exchange, replacement, or warranty "
            "request. If lifecycle cues are visible and no terminal item state is "
            "visible yet, use a lifecycle write tool before or alongside any "
            "monetary adjustment."
        ),
    }


def tool_family_selection_errors(
    payload: Mapping[str, Any], proposal: Any
) -> list[str]:
    call_names = {
        str(call.get("name") or "")
        for call in proposal_tool_calls(proposal)
        if isinstance(call, Mapping)
    }
    if not call_names & MONETARY_ONLY_WRITE_TOOL_NAMES:
        return []
    if call_names & LIFECYCLE_WRITE_TOOL_NAMES:
        return []
    guidance = tool_family_guidance(payload)
    if (
        guidance["active_item_lifecycle_request_detected"] is True
        and guidance["visible_terminal_item_state"] is False
    ):
        return ["refund_tool_family_without_lifecycle_write"]
    return []


def signed_delivery_missing_claim_errors(
    payload: Mapping[str, Any],
    proposal: Any,
) -> list[str]:
    guidance = signed_delivery_missing_claim_guidance(payload)
    blocked_item_ids = set(str(item_id) for item_id in guidance["blocked_item_ids"])
    if not blocked_item_ids:
        return []
    for call in proposal_tool_calls(proposal):
        if not is_confirm_call(call):
            continue
        name = str(call.get("name") or "")
        if name not in {"process_return", "process_refund"}:
            continue
        args = (
            call.get("arguments") if isinstance(call.get("arguments"), Mapping) else {}
        )
        item_id = str(args.get("item_id") or "")
        if item_id in blocked_item_ids:
            return [SIGNED_DELIVERY_PACKAGE_NOT_FOUND_ERROR]
    return []


def signed_delivery_missing_claim_guidance(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    signed_items: dict[str, dict[str, Any]] = {}
    signed_order_ids: set[str] = set()
    signatures: list[str] = []
    for call in extract_tool_calls_with_results(payload):
        if str(call.get("name") or "") != "get_order":
            continue
        result = call.get("result") if isinstance(call.get("result"), Mapping) else {}
        signature = str(result.get("signature_on_file") or "").strip()
        if not signature:
            continue
        order_id = str(result.get("order_id") or "").strip()
        if order_id:
            signed_order_ids.add(order_id)
        signatures.append(signature)
        for item in _order_items(result):
            item_id = str(item.get("item_id") or "").strip()
            if not item_id:
                continue
            signed_items[item_id] = {
                "aliases": sorted(_item_aliases(item)),
                "order_id": order_id,
            }

    blocked: set[str] = set()
    content_missing: set[str] = set()
    user_text = _visible_user_text(payload)
    for item_id, item in signed_items.items():
        aliases = item["aliases"]
        has_content_missing = any(
            _alias_has_content_missing_claim(user_text, alias) for alias in aliases
        )
        has_package_not_found = any(
            _alias_has_package_not_found_claim(user_text, alias) for alias in aliases
        )
        if has_content_missing:
            content_missing.add(item_id)
        if has_package_not_found and not has_content_missing:
            blocked.add(item_id)

    return {
        "active": bool(signed_items and (blocked or content_missing)),
        "rule": (
            "For a signed delivered order, a package-not-found claim for an item "
            "is not a grounded missing-item refund target; a contents-missing item "
            "can remain writable when the user specifically describes that item as "
            "missing from the delivered order."
        ),
        "signature_on_file": signatures[0] if signatures else "",
        "signed_order_ids": sorted(signed_order_ids),
        "blocked_item_ids": sorted(blocked),
        "content_missing_item_ids": sorted(content_missing),
    }


def lifecycle_request_cues(payload: Mapping[str, Any]) -> set[str]:
    text = _visible_dialogue_text(payload)
    return {term for term in LIFECYCLE_REQUEST_CUE_TERMS if term in text}


def visible_terminal_item_state(payload: Mapping[str, Any]) -> bool:
    for call in extract_tool_calls_with_results(payload):
        name = str(call.get("name") or "")
        if name not in LIFECYCLE_WRITE_TOOL_NAMES and name != "cancel_order":
            continue
        if _contains_terminal_item_state(call.get("result")):
            return True
    return False


def proposal_tool_calls(proposal: Any) -> tuple[Mapping[str, Any], ...]:
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


def _visible_dialogue_text(payload: Mapping[str, Any]) -> str:
    return " ".join(
        str(message.get("content") or "").lower()
        for message in visible_conversation(payload)
        if message.get("role") in {"user", "assistant"}
    )


def _visible_user_text(payload: Mapping[str, Any]) -> str:
    return " ".join(
        str(message.get("content") or "").casefold()
        for message in visible_conversation(payload)
        if message.get("role") == "user"
    )


def _order_items(order_result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = order_result.get("items")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _item_aliases(item: Mapping[str, Any]) -> set[str]:
    aliases: set[str] = set()
    for key in ("item_id", "product_id"):
        value = str(item.get(key) or "").casefold().strip()
        if value:
            aliases.add(value)
    product = item.get("product") if isinstance(item.get("product"), Mapping) else {}
    name = str(product.get("name") or item.get("name") or "").casefold()
    words = re.findall(r"[a-z0-9]+", name)
    if words:
        aliases.add(" ".join(words))
    for width in (2, 3):
        for index in range(0, max(0, len(words) - width + 1)):
            aliases.add(" ".join(words[index : index + width]))
    for noun in ("laptop", "phone case", "case"):
        if noun in " ".join(words):
            aliases.add(noun)
    return {alias for alias in aliases if alias}


def _alias_has_content_missing_claim(text: str, alias: str) -> bool:
    if not alias:
        return False
    pattern = re.escape(alias)
    return any(
        re.search(candidate, text)
        for candidate in (
            rf"\b(?:the\s+)?{pattern}\s+(?:is|was|arrived|definitely\s+was)\s+missing\b",
            rf"\bmissing\s+(?:the\s+)?{pattern}\b",
            rf"\b(?:the\s+)?{pattern}\s+missing\s+from\s+(?:the\s+)?(?:box|package|order|shipment)\b",
            rf"\b(?:the\s+)?{pattern}\s+(?:is|was)\s+not\s+in\s+(?:the\s+)?(?:box|package|order|shipment)\b",
        )
    )


def _alias_has_package_not_found_claim(text: str, alias: str) -> bool:
    if not alias:
        return False
    pattern = re.escape(alias)
    return any(
        re.search(candidate, text)
        for candidate in (
            rf"\b(?:can(?:not|'t)|could(?:not|n't)|do\s+not|don't)\s+find\s+(?:the\s+)?{pattern}\b",
            rf"\b(?:can(?:not|'t)|could(?:not|n't))\s+locate\s+(?:the\s+)?{pattern}\b",
            rf"\b(?:did(?:\s+not|n't)|never)\s+receive(?:d)?\s+(?:the\s+)?{pattern}\b",
            rf"\b(?:the\s+)?{pattern}\s+(?:never|did(?:\s+not|n't))\s+arrive(?:d)?\b",
            rf"\b(?:the\s+)?{pattern}\s+(?:is|was)\s+not\s+received\b",
        )
    )


def _contains_terminal_item_state(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in {"status", "item_status"} and isinstance(child, str):
                if child.lower() in {
                    *TERMINAL_ITEM_STATES,
                    "claimed",
                    "store_credit_issued",
                }:
                    return True
            if _contains_terminal_item_state(child):
                return True
        return False
    if isinstance(value, list):
        return any(_contains_terminal_item_state(child) for child in value)
    return False


def _active_warranty_record_matches(
    call: Mapping[str, Any],
    *,
    item_id: str,
    warranty_id: str,
) -> bool:
    args = call.get("arguments") if isinstance(call.get("arguments"), Mapping) else {}
    result = call.get("result") if isinstance(call.get("result"), Mapping) else {}
    result_item = str(result.get("item_id") or args.get("item_id") or "")
    result_warranty = str(result.get("warranty_id") or "")
    status = str(result.get("status") or "").lower()
    is_active = result.get("is_active") is True or status == "active"
    return is_active and result_item == item_id and result_warranty == warranty_id


def _latest_user_explicitly_requests_warranty_claim(payload: Mapping[str, Any]) -> bool:
    for message in reversed(_conversation(payload)):
        if message.get("role") != "user":
            continue
        text = str(message.get("content") or "").lower()
        if any(
            marker in text for marker in ("do not", "don't", "no action", "not yet")
        ):
            return False
        return "warranty" in text and any(
            term in text for term in ("start", "process", "repair", "claim", "go ahead")
        )
    return False


def matching_preview(
    payload: Mapping[str, Any],
    confirm_call: Mapping[str, Any],
    *,
    additional_preview_records: Sequence[Mapping[str, Any]] = (),
) -> Mapping[str, Any] | None:
    preview, _error = _matching_preview_verdict(
        payload,
        confirm_call,
        additional_preview_records=additional_preview_records,
    )
    return preview


def _matching_preview_verdict(
    payload: Mapping[str, Any],
    confirm_call: Mapping[str, Any],
    *,
    additional_preview_records: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any] | None, str | None]:
    confirm_name = str(confirm_call.get("name") or "")
    confirm_args = (
        confirm_call.get("arguments")
        if isinstance(confirm_call.get("arguments"), Mapping)
        else {}
    )
    additional_candidates = [
        preview
        for preview in additional_preview_records
        if is_preview_call(preview) and preview.get("name") == confirm_name
    ]
    if additional_candidates:
        return _select_matching_preview(confirm_args, additional_candidates)
    visible_candidates = [
        preview
        for preview in preview_records(payload)
        if preview.get("name") == confirm_name
    ]
    return _select_matching_preview(confirm_args, visible_candidates)


def _select_matching_preview(
    confirm_args: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any] | None, str | None]:
    if not candidates:
        return None, "confirm_without_preview"
    confirm_tokens = id_like_tokens(confirm_args)
    if not confirm_tokens:
        if len(candidates) == 1:
            return candidates[0], None
        return None, "ambiguous_confirm_preview_binding"
    bound_candidates: list[Mapping[str, Any]] = []
    for preview in candidates:
        preview_args = (
            preview.get("arguments")
            if isinstance(preview.get("arguments"), Mapping)
            else {}
        )
        preview_result = (
            preview.get("result") if isinstance(preview.get("result"), Mapping) else {}
        )
        preview_tokens = id_like_tokens(
            {"arguments": preview_args, "result": preview_result}
        )
        if preview_tokens and not confirm_tokens.isdisjoint(preview_tokens):
            bound_candidates.append(preview)
    if len(bound_candidates) == 1:
        return bound_candidates[0], None
    if len(bound_candidates) > 1:
        return None, "ambiguous_confirm_preview_binding"
    return None, "confirm_without_preview"


def _preview_candidates(
    payload: Mapping[str, Any],
    additional_preview_records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = list(preview_records(payload))
    rows.extend(call for call in additional_preview_records if is_preview_call(call))
    return rows


def is_preview_call(call: Mapping[str, Any]) -> bool:
    name = str(call.get("name") or "")
    args = call.get("arguments") if isinstance(call.get("arguments"), Mapping) else {}
    result = call.get("result") if isinstance(call.get("result"), Mapping) else {}
    return (
        name in PREVIEW_CONFIRM_TOOL_NAMES
        and args.get("confirm") is not True
        and str(result.get("status") or "").lower() == "preview"
    )


def is_confirm_call(call: Mapping[str, Any]) -> bool:
    name = str(call.get("name") or "")
    args = call.get("arguments") if isinstance(call.get("arguments"), Mapping) else {}
    return name in PREVIEW_CONFIRM_TOOL_NAMES and args.get("confirm") is True


def is_from_preview_placeholder(value: Any) -> bool:
    return isinstance(value, Mapping) and isinstance(value.get("$from_preview"), str)


def _resolve_deferred_preview_args(
    confirm_call: Mapping[str, Any],
    preview: Mapping[str, Any],
) -> dict[str, Any]:
    args = (
        dict(confirm_call.get("arguments"))
        if isinstance(confirm_call.get("arguments"), Mapping)
        else {}
    )
    preview_result = (
        preview.get("result") if isinstance(preview.get("result"), Mapping) else {}
    )
    alias_to_slot = {
        alias: slot
        for slot, aliases in CONFIRM_SLOT_ARG_ALIASES.items()
        for alias in aliases
    }
    errors: list[str] = []
    resolved_slots: list[str] = []
    for arg_name, value in list(args.items()):
        if not is_from_preview_placeholder(value):
            continue
        slot = alias_to_slot.get(arg_name)
        requested_slot = str(value["$from_preview"])
        if slot is None:
            errors.append(f"{arg_name}_preview_binding_unsupported")
            continue
        if requested_slot != slot:
            errors.append(f"{slot}_preview_binding_slot_mismatch")
            continue
        if requested_slot not in preview_result:
            errors.append(f"{slot}_preview_value_missing")
            continue
        args[arg_name] = preview_result[requested_slot]
        resolved_slots.append(slot)
    resolved_call = _copy_call(confirm_call)
    resolved_call["arguments"] = args
    return {
        "errors": errors,
        "resolved_call": resolved_call,
        "resolved_slots": resolved_slots,
    }


def _copy_call(call: Mapping[str, Any]) -> dict[str, Any]:
    args = call.get("arguments") if isinstance(call.get("arguments"), Mapping) else {}
    return {
        "name": str(call.get("name") or ""),
        "arguments": dict(args),
    }


def _conversation(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = payload.get("conversation")
    if not isinstance(raw, list):
        return []
    return [message for message in raw if isinstance(message, Mapping)]


def _numeric_token_variants(token: str) -> set[str]:
    lowered = token.lower()
    variants = {lowered}
    if lowered.startswith("$"):
        variants.add(lowered[1:])
    elif lowered and lowered[0].isdigit() and not lowered.endswith("%"):
        variants.add(f"${lowered}")
    return variants

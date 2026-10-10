"""Visible-data Shopping residual gates for STATE-Bench pilots."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from papc.certspec.state_bench import request_cues

ID_LIKE_RE = re.compile(
    r"\b(?:SP|CI|CART|VAR)[-_][A-Za-z0-9_]+\b"
    r"|\bshop_[A-Za-z0-9_]+\b"
    r"|\b(?:SAVE\d+|WELCOME\d+)\b",
    flags=re.IGNORECASE,
)
NUMERIC_LIKE_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?:"
    r"\$?\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?%?(?![A-Za-z0-9_])"
    r"|"
    r"\$?\d+(?:\.\d{1,2})?%?(?![A-Za-z0-9_-]|,\d)"
    r")"
)
EARNED_POINTS_RE = re.compile(
    r"\b(\d{1,3}(?:,\d{3})+|\d+)\s+points\b", flags=re.IGNORECASE
)
POLICY_TERMS = (
    "backorder",
    "bonus",
    "bundle",
    "cost",
    "discount",
    "fee",
    "loyalty",
    "policy",
    "price",
    "promo",
    "quote",
    "refund",
    "return",
    "shipping",
)
COMPARISON_TERMS = ("alternative", "cheaper", "compare", "option", "tradeoff")
COMPARISON_COVER_TERMS = (
    *COMPARISON_TERMS,
    "both",
    "bottom line",
    "path",
    "versus",
    "vs",
)
CONFIRMATION_TERMS = (
    "approval",
    "before",
    "breakdown",
    "check out",
    "checkout",
    "confirm",
    "consent",
    "finalise",
    "finalize",
    "total",
)
DIALOGUE_TERMS = ("decline", "explain", "mention", "recommend", "tell", "warn")
DIALOGUE_COVER_TERMS = (
    *DIALOGUE_TERMS,
    "cleanup",
    "not applied",
    "not qualify",
    "qualify",
    "won't qualify",
    "will not qualify",
)
DIALOGUE_FAMILY_TERMS = {
    "policy_quote": POLICY_TERMS,
    "comparison": COMPARISON_TERMS,
    "confirmation": CONFIRMATION_TERMS,
    "dialogue": DIALOGUE_TERMS,
}
DIALOGUE_FAMILY_COVER_TERMS = {
    "policy_quote": POLICY_TERMS,
    "comparison": COMPARISON_COVER_TERMS,
    "confirmation": CONFIRMATION_TERMS,
    "dialogue": DIALOGUE_COVER_TERMS,
}
FINAL_SUMMARY_TERMS = (
    "breakdown",
    "checkout",
    "check out",
    "final",
    "finalize",
    "finalise",
)
TOTAL_TERMS = ("total", *FINAL_SUMMARY_TERMS)
DISCOUNT_TERMS = ("discount", "promo", "savings", *FINAL_SUMMARY_TERMS)
FEE_TERMS = ("fee", "gift wrap", "wrap", *FINAL_SUMMARY_TERMS)
HINT_EXTRACTOR_VERSION = "shopping_hint_extractor_v1.1_user_turn_only"
SUMMARY_FAMILIES_BY_WRITE_TOOL = {
    "add_to_cart": frozenset({"total", "earned_points"}),
    "update_cart_item": frozenset({"total", "earned_points"}),
    "remove_from_cart": frozenset({"total", "earned_points"}),
    "apply_promo": frozenset({"discount", "total", "earned_points"}),
    "remove_promo": frozenset({"discount", "total", "earned_points"}),
    "redeem_loyalty_points": frozenset({"discount", "total", "earned_points"}),
    "cancel_loyalty_redemption": frozenset({"discount", "total", "earned_points"}),
    "set_shipping_option": frozenset({"fee", "total", "earned_points"}),
}
READ_BEFORE_MESSAGE_CONTEXT_KEY = "_papc_read_before_message_probe_context"
CATALOG_PRICE_NOT_PROMO_UNIT = "dialogue.shopping.catalog_price_not_promo"
BACKORDER_DISCLOSURE_UNIT = "dialogue.shopping.backorder_disclosure"
CURRENT_CART_TOTAL_PRICE_DROP_UNIT = "dialogue.shopping.current_cart_total_price_drop"
PRODUCT_VERIFICATION_LOOKUP_UNIT = "dialogue.shopping.product_verification_lookup"
QUANTITY_CAP_STOCK_FINAL_QUANTITY_UNIT = (
    "dialogue.shopping.quantity_cap_stock_final_quantity"
)
UNSUPPORTED_SAVE_FOR_LATER_LOOKUP_UNIT = (
    "dialogue.shopping.unsupported_save_for_later_lookup"
)


def extract_tool_calls_with_results(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for message in (
        payload.get("conversation", [])
        if isinstance(payload.get("conversation"), list)
        else []
    ):
        if not isinstance(message, Mapping) or not isinstance(
            message.get("tool_calls"), list
        ):
            continue
        for call in message["tool_calls"]:
            if isinstance(call, Mapping):
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
    for message in (
        payload.get("conversation", [])
        if isinstance(payload.get("conversation"), list)
        else []
    ):
        if not isinstance(message, Mapping):
            continue
        entry = {
            "role": message.get("role"),
            "content": message.get("content"),
        }
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            entry["tool_calls"] = [
                {
                    "name": call.get("name"),
                    "arguments": call.get("arguments"),
                    "result": call.get("result"),
                }
                for call in calls
                if isinstance(call, Mapping)
            ]
        visible.append(entry)
    return visible


def visible_evidence_tokens(payload: Mapping[str, Any]) -> set[str]:
    blob = json.dumps(
        {
            "visible_conversation": visible_conversation(payload),
            "read_before_message_probe_context": read_before_message_visible_context(
                payload
            ),
        },
        ensure_ascii=False,
    )
    return {token.lower() for token in ID_LIKE_RE.findall(blob)}


def id_like_tokens(value: Any) -> set[str]:
    return {
        token for token in ID_LIKE_RE.findall(json.dumps(value, ensure_ascii=False))
    }


def visible_numeric_tokens(payload: Mapping[str, Any]) -> set[str]:
    blob = json.dumps(
        {
            "visible_conversation": visible_conversation(payload),
            "read_before_message_probe_context": read_before_message_visible_context(
                payload
            ),
        },
        ensure_ascii=False,
    )
    return numeric_like_tokens(blob)


def read_before_message_visible_context(payload: Mapping[str, Any]) -> dict[str, Any]:
    context = payload.get(READ_BEFORE_MESSAGE_CONTEXT_KEY)
    if not isinstance(context, Mapping):
        return {}
    return {
        "schema": context.get("schema"),
        "enabled": bool(context.get("enabled")),
        "open_dialogue_families": [
            str(family)
            for family in context.get("open_dialogue_families", [])
            if isinstance(family, str)
        ],
        "probe_results": [
            {
                "family": row.get("family"),
                "name": row.get("name"),
                "result": row.get("result"),
            }
            for row in context.get("probe_results", [])
            if isinstance(row, Mapping)
        ],
    }


def read_before_message_open_dialogue_families(
    payload: Mapping[str, Any],
) -> tuple[str, ...]:
    context = read_before_message_visible_context(payload)
    families: list[str] = []
    for raw in context.get("open_dialogue_families", []):
        family = str(raw).strip()
        if family.startswith("dialogue.shopping."):
            family = family.removeprefix("dialogue.shopping.")
        if family:
            families.append(family)
    return tuple(dict.fromkeys(families))


def selector_protocol_dialogue_units(payload: Mapping[str, Any]) -> frozenset[str]:
    units: set[str] = set()
    if _catalog_price_not_promo_frontier(payload) is not None:
        units.add(CATALOG_PRICE_NOT_PROMO_UNIT)
    if _backorder_disclosure_missed(payload):
        units.add(BACKORDER_DISCLOSURE_UNIT)
    if _quantity_cap_stock_final_quantity_gap(payload):
        units.add(QUANTITY_CAP_STOCK_FINAL_QUANTITY_UNIT)
    if _current_cart_total_price_drop_gap(payload):
        units.add(CURRENT_CART_TOTAL_PRICE_DROP_UNIT)
    if _product_verification_missed_frontier(payload) is not None:
        units.add(PRODUCT_VERIFICATION_LOOKUP_UNIT)
    if _unsupported_save_for_later_frontier(payload) is not None:
        units.add(UNSUPPORTED_SAVE_FOR_LATER_LOOKUP_UNIT)
    return frozenset(units)


def selector_frontier_position(
    payload: Mapping[str, Any],
    *,
    open_dialogue_units: tuple[str, ...],
    candidate_class: str,
) -> tuple[int, int | None] | None:
    del candidate_class
    units = set(open_dialogue_units)
    if CATALOG_PRICE_NOT_PROMO_UNIT in units:
        return _catalog_price_not_promo_frontier(payload)
    if BACKORDER_DISCLOSURE_UNIT in units:
        return _terminal_frontier(payload)
    if QUANTITY_CAP_STOCK_FINAL_QUANTITY_UNIT in units:
        return _quantity_cap_stock_frontier(payload)
    if CURRENT_CART_TOTAL_PRICE_DROP_UNIT in units:
        return _current_cart_total_frontier(payload)
    if PRODUCT_VERIFICATION_LOOKUP_UNIT in units:
        return _product_verification_missed_frontier(payload)
    if UNSUPPORTED_SAVE_FOR_LATER_LOOKUP_UNIT in units:
        return _unsupported_save_for_later_frontier(payload)
    return None


def _terminal_frontier(payload: Mapping[str, Any]) -> tuple[int, None]:
    messages = visible_conversation(payload)
    task_done_index = _task_done_message_index(messages)
    if task_done_index is not None:
        for index in range(task_done_index - 1, -1, -1):
            if str(messages[index].get("role") or "") == "assistant":
                return index, None
        return task_done_index, None
    return len(messages), None


def _task_done_message_index(messages: list[dict[str, Any]]) -> int | None:
    for index, message in enumerate(messages):
        if str(message.get("role") or "") != "user":
            continue
        if "[TASK_DONE]" in str(message.get("content") or ""):
            return index
    return None


def _catalog_price_not_promo_frontier(
    payload: Mapping[str, Any],
) -> tuple[int, int] | None:
    messages = visible_conversation(payload)
    prior_user_requested_price_vs_promo = False
    for index, message in enumerate(messages):
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")
        if role == "user" and _user_asks_catalog_price_not_promo(content):
            prior_user_requested_price_vs_promo = True
            continue
        if role != "assistant" or not prior_user_requested_price_vs_promo:
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        saw_product_price_drop = False
        for call_index, call in enumerate(calls):
            if not isinstance(call, Mapping):
                continue
            name = str(call.get("name") or "")
            if _call_shows_catalog_price_drop(call):
                saw_product_price_drop = True
            if name == "get_promotions" and saw_product_price_drop:
                return index, call_index
    return None


def _user_asks_catalog_price_not_promo(content: str) -> bool:
    text = str(content or "").casefold()
    return (
        "promo" in text
        and "price" in text
        and ("catalog" in text or "promo code" in text)
        and any(marker in text for marker in ("before adding", "before add", "wait"))
    )


def _call_shows_catalog_price_drop(call: Mapping[str, Any]) -> bool:
    result = call.get("result")
    if not isinstance(result, Mapping):
        return False
    if _mapping_has_price_drop_fields(result):
        return True
    products = result.get("products")
    if isinstance(products, list):
        return any(
            isinstance(product, Mapping) and _mapping_has_price_drop_fields(product)
            for product in products
        )
    return False


def _mapping_has_price_drop_fields(value: Mapping[str, Any]) -> bool:
    return value.get("previous_price") not in {None, "", 0} and value.get(
        "price"
    ) not in {None, ""}


def _product_verification_missed_frontier(
    payload: Mapping[str, Any],
) -> tuple[int, None] | None:
    messages = visible_conversation(payload)
    for index, message in enumerate(messages):
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")
        if role != "user" or not _user_asks_product_verification(content):
            continue
        for assistant_index in range(index + 1, len(messages)):
            assistant = messages[assistant_index]
            assistant_role = str(assistant.get("role") or "")
            if assistant_role == "user":
                break
            if assistant_role != "assistant":
                continue
            calls = assistant.get("tool_calls")
            assistant_content = str(assistant.get("content") or "")
            if (
                isinstance(calls, list)
                and calls
                and _all_product_searches_empty(calls)
                and _assistant_reports_product_not_found(assistant_content)
            ):
                return assistant_index, None
            break
    return None


def _backorder_disclosure_missed(payload: Mapping[str, Any]) -> bool:
    messages = visible_conversation(payload)
    saw_backorder_request = False
    saw_backorder_policy = False
    saw_product_details = False
    saw_cannot_verify_backorder = False
    for message in messages:
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")
        if role == "user" and _user_asks_backorder(content):
            saw_backorder_request = True
        if role != "assistant" or not saw_backorder_request:
            continue
        if _assistant_cannot_verify_backorder(content):
            saw_cannot_verify_backorder = True
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            name = str(call.get("name") or "")
            if name == "get_product_details":
                saw_product_details = True
            if name == "get_policies":
                args = call.get("arguments")
                if isinstance(args, Mapping) and args.get("topic") == "backorder":
                    saw_backorder_policy = True
    return (
        saw_backorder_request
        and saw_backorder_policy
        and saw_cannot_verify_backorder
        and not saw_product_details
    )


def _quantity_cap_stock_final_quantity_gap(payload: Mapping[str, Any]) -> bool:
    messages = visible_conversation(payload)
    if not any(
        str(message.get("role") or "") == "user"
        and _user_asks_quantity_cap_stock_without_write(
            str(message.get("content") or "")
        )
        for message in messages
    ):
        return False
    saw_cart = False
    saw_quantity_policy = False
    saw_stock_details = False
    for message in messages:
        if str(message.get("role") or "") != "assistant":
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            name = str(call.get("name") or "")
            args = call.get("arguments")
            result = call.get("result")
            if name == "get_cart":
                saw_cart = True
            if (
                name == "get_policies"
                and isinstance(args, Mapping)
                and args.get("topic") == "quantity_limit"
            ):
                saw_quantity_policy = True
            if (
                name == "get_product_details"
                and isinstance(result, Mapping)
                and result.get("stock_quantity") not in {None, ""}
            ):
                saw_stock_details = True
    return saw_cart and saw_quantity_policy and saw_stock_details


def _quantity_cap_stock_frontier(
    payload: Mapping[str, Any],
) -> tuple[int, None] | None:
    messages = visible_conversation(payload)
    for index, message in enumerate(messages):
        if str(message.get("role") or "") != "user":
            continue
        if not _user_asks_quantity_cap_stock_without_write(
            str(message.get("content") or "")
        ):
            continue
        for candidate_index in range(index + 1, len(messages)):
            candidate = messages[candidate_index]
            if str(candidate.get("role") or "") == "assistant":
                return candidate_index, None
    return None


def _current_cart_total_price_drop_gap(payload: Mapping[str, Any]) -> bool:
    messages = visible_conversation(payload)
    saw_current_total_request = False
    for message in messages:
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")
        if role == "user" and _user_asks_current_cart_total(content):
            saw_current_total_request = True
        if role != "assistant" or not saw_current_total_request:
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            if str(call.get("name") or "") != "get_cart":
                continue
            result = call.get("result")
            if isinstance(result, Mapping) and _cart_result_has_stale_total(result):
                return True
    return False


def _current_cart_total_frontier(
    payload: Mapping[str, Any],
) -> tuple[int, None] | None:
    messages = visible_conversation(payload)
    for index, message in enumerate(messages):
        if str(message.get("role") or "") != "user":
            continue
        if not _user_asks_current_cart_total(str(message.get("content") or "")):
            continue
        for candidate_index in range(index + 1, len(messages)):
            candidate = messages[candidate_index]
            if str(candidate.get("role") or "") == "assistant":
                return candidate_index, None
    return None


def _unsupported_save_for_later_frontier(
    payload: Mapping[str, Any],
) -> tuple[int, None] | None:
    messages = visible_conversation(payload)
    for index, message in enumerate(messages):
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")
        if role != "user" or not _user_asks_save_for_later(content):
            continue
        for assistant_index in range(index + 1, len(messages)):
            assistant = messages[assistant_index]
            assistant_role = str(assistant.get("role") or "")
            if assistant_role == "user":
                break
            if assistant_role != "assistant":
                continue
            calls = assistant.get("tool_calls")
            if isinstance(calls, list) and calls:
                return None
            assistant_content = str(assistant.get("content") or "")
            if _assistant_declines_save_for_later(assistant_content):
                return assistant_index, None
            break
    return None


def _user_asks_current_cart_total(content: str) -> bool:
    text = str(content or "").casefold()
    return (
        any(marker in text for marker in ("right now", "current", "currently", "now"))
        and "cart" in text
        and any(marker in text for marker in ("total", "price", "cost"))
    )


def _cart_result_has_stale_total(result: Mapping[str, Any]) -> bool:
    line_total_sum = 0.0
    saw_line = False
    items = result.get("items")
    if not isinstance(items, list):
        return False
    for item in items:
        if not isinstance(item, Mapping):
            continue
        line_total = _money_number(item.get("line_total"))
        if line_total is None:
            unit_price = _money_number(item.get("unit_price"))
            quantity = _money_number(item.get("quantity"))
            if unit_price is None or quantity is None:
                continue
            line_total = unit_price * quantity
        line_total_sum += line_total
        saw_line = True
    if not saw_line:
        return False
    subtotal = _money_number(result.get("subtotal"))
    total = _money_number(result.get("total"))
    return any(
        value is not None and abs(value - line_total_sum) > 0.001
        for value in (subtotal, total)
    )


def _money_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    text = str(value or "").strip().replace("$", "").replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _user_asks_quantity_cap_stock_without_write(content: str) -> bool:
    text = str(content or "").casefold()
    return (
        "cap" in text
        and "stock" in text
        and "quantity" in text
        and any(marker in text for marker in ("do not change", "don't change"))
        and "final quantity" in text
    )


def _user_asks_backorder(content: str) -> bool:
    text = str(content or "").casefold()
    return "backorder" in text or "backordered" in text


def _assistant_cannot_verify_backorder(content: str) -> bool:
    text = str(content or "").replace("\u2019", "'").casefold()
    return "backorder" in text and any(
        marker in text
        for marker in (
            "can't verify",
            "cannot verify",
            "can't confirm",
            "cannot confirm",
            "doesn't appear",
            "does not appear",
            "can't find",
            "couldn't find",
            "not able to confirm",
        )
    )


def _user_asks_product_verification(content: str) -> bool:
    text = str(content or "").casefold()
    if not any(marker in text for marker in ("verify", "check", "confirm")):
        return False
    return any(marker in text for marker in ("exists", "exist", "compatible", "fits"))


def _all_product_searches_empty(calls: list[Any]) -> bool:
    saw_search = False
    for call in calls:
        if not isinstance(call, Mapping):
            continue
        if str(call.get("name") or "") != "search_products":
            continue
        saw_search = True
        result = call.get("result")
        if not isinstance(result, Mapping):
            return False
        products = result.get("products")
        if isinstance(products, list) and products:
            return False
        total_found = result.get("total_found")
        if total_found not in {None, 0, "0"}:
            return False
    return saw_search


def _assistant_reports_product_not_found(content: str) -> bool:
    text = str(content or "").replace("\u2019", "'").casefold()
    return any(
        marker in text
        for marker in (
            "couldn't find",
            "could not find",
            "can't find",
            "cannot find",
            "no matching",
            "no match",
            "not seeing",
            "doesn't appear",
        )
    )


def _user_asks_save_for_later(content: str) -> bool:
    text = str(content or "").casefold()
    return (
        ("wishlist" in text or "save for later" in text or "save it for later" in text)
        and any(marker in text for marker in ("save", "remember", "later"))
        and not any(marker in text for marker in ("cart", "buy now", "purchase now"))
    )


def _assistant_declines_save_for_later(content: str) -> bool:
    text = str(content or "").replace("\u2019", "'").casefold()
    if not ("wishlist" in text or "save" in text or "later" in text):
        return False
    return any(
        marker in text
        for marker in (
            "can't",
            "cannot",
            "do not have",
            "don't have",
            "not able",
            "not support",
            "can't create",
        )
    )


def numeric_like_tokens(value: Any) -> set[str]:
    tokens: set[str] = set()
    for token in NUMERIC_LIKE_RE.findall(json.dumps(value, ensure_ascii=False)):
        tokens.update(_numeric_token_variants(token))
    return tokens


def numeric_tokens_not_in_visible_evidence(
    value: Any,
    visible_tokens: set[str] | frozenset[str],
) -> set[str]:
    visible = {token.lower() for token in visible_tokens}
    missing: set[str] = set()
    for token in NUMERIC_LIKE_RE.findall(json.dumps(value, ensure_ascii=False)):
        variants = {variant.lower() for variant in _numeric_token_variants(token)}
        if not variants.intersection(visible):
            missing.add(token)
    return missing


def _numeric_token_variants(token: str) -> set[str]:
    lowered = token.lower()
    comma_stripped = lowered.replace(",", "")
    variants = {lowered, comma_stripped}
    percent_stripped = comma_stripped.removesuffix("%")
    if percent_stripped != comma_stripped:
        variants.add(percent_stripped)
    if comma_stripped.startswith("$"):
        variants.add(comma_stripped[1:])
    elif (
        comma_stripped
        and comma_stripped[0].isdigit()
        and not comma_stripped.endswith("%")
    ):
        variants.add(f"${comma_stripped}")
    return variants


def final_summary_numeric_obligation_errors(
    payload: Mapping[str, Any],
    assistant_message: str,
    *,
    exempt_families: set[str] | frozenset[str] | None = None,
) -> list[str]:
    """Require visible numeric summary values when user asks for a final summary."""

    required = final_summary_required_numeric_tokens(payload)
    if not required:
        return []
    exempt = {str(family) for family in exempt_families or set()}
    message_tokens = numeric_like_tokens(assistant_message)
    errors: list[str] = []
    for family, tokens in required.items():
        if family in exempt:
            continue
        if (
            family == "discount"
            and _loyalty_redemption_cancel_message_satisfies_discount(
                assistant_message,
                tokens,
            )
        ):
            continue
        if tokens and not (tokens & message_tokens):
            errors.append(f"message_missing_visible_{family}_numeric_summary")
    return errors


def _loyalty_redemption_cancel_message_satisfies_discount(
    assistant_message: str,
    required_discount_tokens: set[str],
) -> bool:
    lowered = str(assistant_message or "").lower()
    if "loyalty" not in lowered or "point" not in lowered:
        return False
    if not any(
        marker in lowered
        for marker in (
            "credited back",
            "points credited",
            "redemption was canceled",
            "redemption was cancelled",
            "no loyalty redemption",
            "not currently applied",
        )
    ):
        return False
    message_tokens = numeric_like_tokens(assistant_message)
    for token in required_discount_tokens:
        raw = str(token).lower().replace("$", "").replace(",", "")
        if not raw.isdigit():
            continue
        points_equivalent = str(int(raw) * 100)
        comma_points = f"{int(raw) * 100:,}"
        if points_equivalent in message_tokens or comma_points in message_tokens:
            return True
    return False


def final_summary_required_numeric_tokens(
    payload: Mapping[str, Any],
) -> dict[str, set[str]]:
    user_text = " ".join(
        str(message.get("content") or "").lower()
        for message in payload.get("conversation", [])
        if isinstance(message, Mapping) and message.get("role") == "user"
    )
    if not user_text:
        return {}
    fields = _latest_visible_numeric_fields(payload)
    requirements: dict[str, set[str]] = {}
    if any(term in user_text for term in TOTAL_TERMS) and fields.get("total"):
        requirements["total"] = fields["total"]
    if any(term in user_text for term in DISCOUNT_TERMS) and fields.get("discount"):
        requirements["discount"] = fields["discount"]
    if any(term in user_text for term in FEE_TERMS) and fields.get("fee"):
        requirements["fee"] = fields["fee"]
    if "points" in user_text and "earn" in user_text and fields.get("earned_points"):
        requirements["earned_points"] = fields["earned_points"]
    return requirements


def _latest_visible_numeric_fields(payload: Mapping[str, Any]) -> dict[str, set[str]]:
    fields: dict[str, set[str]] = {}
    visible_messages = visible_conversation(payload)
    for message in visible_messages:
        for call in (
            message.get("tool_calls", [])
            if isinstance(message.get("tool_calls"), list)
            else []
        ):
            if not isinstance(call, Mapping) or not isinstance(
                call.get("result"), Mapping
            ):
                continue
            result = call["result"]
            _capture_field(fields, "total", result, ("cart_total", "total"))
            _capture_field(
                fields,
                "discount",
                result,
                ("discount_amount", "estimated_discount", "loyalty_discount"),
            )
            _capture_field(fields, "fee", result, ("gift_wrap_fee", "shipping_cost"))
        content = str(message.get("content") or "")
        lowered = content.lower()
        if "gift wrap fee" in lowered:
            _capture_text_family(fields, "fee", content)
        if "promo discount" in lowered or "discount" in lowered:
            _capture_text_family(fields, "discount", content)
        if "total" in lowered:
            _capture_text_family(fields, "total", content)
        if "points" in lowered and ("earn" in lowered or "earned" in lowered):
            for match in EARNED_POINTS_RE.findall(content):
                fields["earned_points"] = _numeric_token_variants(match)
    return fields


def _capture_field(
    fields: dict[str, set[str]],
    family: str,
    result: Mapping[str, Any],
    names: tuple[str, ...],
) -> None:
    for name in names:
        value = result.get(name)
        if isinstance(value, int | float | str) and str(value).strip() not in {
            "",
            "0",
            "0.0",
            "$0",
        }:
            fields[family] = numeric_like_tokens(value)


def _capture_text_family(
    fields: dict[str, set[str]], family: str, content: str
) -> None:
    if family in fields:
        return
    terms = {
        "discount": ("discount", "promo", "savings"),
        "fee": ("fee", "gift wrap"),
        "total": ("total",),
    }.get(family, (family,))
    for segment in re.split(r"[\n.;]", content):
        if not any(term in segment.lower() for term in terms):
            continue
        tokens = numeric_like_tokens(segment)
        if tokens:
            fields[family] = tokens
            return


def open_dialogue_obligation_families(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Visible-only communicative obligations that can be discharged by a message."""

    open_by_family: dict[str, int] = {}
    messages = payload.get("conversation", [])
    for turn_index, message in enumerate(
        messages if isinstance(messages, list) else []
    ):
        if not isinstance(message, Mapping):
            continue
        content = str(message.get("content") or "")
        if message.get("role") == "user":
            for sentence in request_cues.sentences(content):
                for family in _dialogue_families_requested_by_user_sentence(sentence):
                    open_by_family[family] = turn_index
        elif message.get("role") == "assistant":
            for family in tuple(open_by_family):
                if message_covers_open_family(content, (family,)):
                    open_by_family.pop(family, None)
    families: list[str] = list(open_by_family)
    families.extend(read_before_message_open_dialogue_families(payload))
    if assistant_final_cart_confirmation_missing_total(payload):
        families.append("confirmation")
    return tuple(dict.fromkeys(families))


def dialogue_obligation_hint_families(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Visible user-requested dialogue families, regardless of later satisfaction."""

    families: list[str] = []
    messages = payload.get("conversation", [])
    for message in messages if isinstance(messages, list) else []:
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        for sentence in request_cues.sentences(str(message.get("content") or "")):
            families.extend(_dialogue_families_requested_by_user_sentence(sentence))
    families.extend(read_before_message_open_dialogue_families(payload))
    if assistant_final_cart_confirmation_missing_total(payload):
        families.append("confirmation")
    return tuple(dict.fromkeys(families))


def assistant_final_cart_confirmation_missing_total(payload: Mapping[str, Any]) -> bool:
    messages = visible_conversation(payload)
    if not messages or messages[-1].get("role") != "assistant":
        return False
    assistant_message = str(messages[-1].get("content") or "")
    lowered = assistant_message.lower()
    if not any(term in lowered for term in ("cart", "shipping", "checkout", "done")):
        return False
    required_total = _latest_visible_numeric_fields(payload).get("total")
    if not required_total:
        return False
    message_tokens = numeric_like_tokens(assistant_message)
    return not bool(required_total & message_tokens)


def _dialogue_families_requested_by_user_sentence(sentence: str) -> tuple[str, ...]:
    lowered = sentence.lower()
    if _is_dialogue_acknowledgement(lowered):
        return ()
    families: list[str] = []
    for family, terms in DIALOGUE_FAMILY_TERMS.items():
        if any(term in lowered for term in terms):
            families.append(family)
    return tuple(families)


def _is_dialogue_acknowledgement(lowered_sentence: str) -> bool:
    return any(
        phrase in lowered_sentence
        for phrase in (
            "that's the comparison i needed",
            "that is the comparison i needed",
            "that's exactly what i needed",
            "that is exactly what i needed",
            "that works",
            "that covers it",
            "we're good",
            "i just wanted the verified comparison",
            "i only wanted the policy explanation",
            "i just needed to understand",
        )
    )


def summary_families_touched_by_tool_calls(
    calls: Any,
) -> frozenset[str]:
    """Numeric summary families invalidated by the proposed write calls."""

    families: set[str] = set()
    if isinstance(calls, Mapping):
        iterable = proposal_tool_calls(calls)
    elif isinstance(calls, (list, tuple)):
        iterable = tuple(call for call in calls if isinstance(call, Mapping))
    else:
        iterable = ()
    for call in iterable:
        name = str(call.get("name") or "")
        families.update(SUMMARY_FAMILIES_BY_WRITE_TOOL.get(name, frozenset()))
        args = (
            call.get("arguments") if isinstance(call.get("arguments"), Mapping) else {}
        )
        if name == "update_cart_item" and any(
            str(key).lower() in {"gift_wrap", "giftwrap", "gift_wrap_enabled"}
            for key in args
        ):
            families.update({"fee", "total", "earned_points"})
    return frozenset(families)


def message_only_dialogue_errors(
    payload: Mapping[str, Any],
    assistant_message: str,
) -> list[str]:
    """Fail closed unless a message-only residual has a visible open obligation."""

    if not assistant_message.strip():
        return []
    families = open_dialogue_obligation_families(payload)
    if not families:
        required_numeric = final_summary_required_numeric_tokens(payload)
        if required_numeric and not final_summary_numeric_obligation_errors(
            payload,
            assistant_message,
        ):
            return []
        return ["message_without_open_dialogue_obligation"]
    if not message_covers_open_family(assistant_message, families):
        return ["message_does_not_discharge_open_dialogue_obligation"]
    return []


def message_covers_open_family(
    assistant_message: str, families: tuple[str, ...]
) -> bool:
    lowered = assistant_message.lower()
    return any(
        any(term in lowered for term in DIALOGUE_FAMILY_COVER_TERMS.get(family, ()))
        for family in families
    )


def proposal_tool_calls(proposal: Any) -> tuple[Mapping[str, Any], ...]:
    calls = getattr(proposal, "tool_calls", ())
    if isinstance(calls, tuple):
        return calls
    if isinstance(calls, list):
        return tuple(call for call in calls if isinstance(call, Mapping))
    if isinstance(proposal, Mapping) and isinstance(proposal.get("tool_calls"), list):
        return tuple(
            call for call in proposal["tool_calls"] if isinstance(call, Mapping)
        )
    return ()

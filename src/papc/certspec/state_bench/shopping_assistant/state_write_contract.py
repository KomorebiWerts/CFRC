"""STATE-Bench shopping projection into shared RCBE state-write contracts.

The shopping surface is a mostly reversible cart-edit surface: cart item edits,
promo application/removal, loyalty redemption/cancel, and shipping-option edits
all expose an inverse or in-place update path in the tool surface. This adapter
keeps those rules under ``src/papc/certspec`` so pilot scripts can route
proposals through the shared certspec kernel instead of duplicating masks.

No task ids, scorer requirements, concrete product ids, cart ids, promo codes,
or fixed tool arguments are consumed by this module. Runtime evidence is reduced
to abstract unit families plus occurrence-free proposal footprints.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from papc.certspec.reversibility import (
    DEFAULT_RESTORATION_COST,
    WriteUnitSignal,
    restoration_cost_from_signal,
)
from papc.certspec.state_bench import request_cues
from papc.certspec.state_bench.shopping_assistant import residual_gates
from papc.certspec.state_write_contract import (
    Obligation,
    ProposedTargetAction,
    StateWriteContract,
)

SURFACE = "state_bench_shopping_assistant"
READ_BEFORE_MESSAGE_CONTEXT_KEY = "_papc_read_before_message_probe_context"
CURRENT_CART_TOTAL_DIALOGUE_FAMILY = "dialogue.shopping.current_cart_total_price_drop"
PRICE_REFRESH_FAMILY = "shopping.cart.price_refresh"

WRITE_TOOL_NAMES = frozenset(
    {
        "add_to_cart",
        "update_cart_item",
        "remove_from_cart",
        "apply_promo",
        "remove_promo",
        "redeem_loyalty_points",
        "cancel_loyalty_redemption",
        "set_shipping_option",
    }
)

REVERSIBLE_TOOL_PAIRS: tuple[tuple[str, str], ...] = (
    ("add_to_cart", "remove_from_cart"),
    ("apply_promo", "remove_promo"),
    ("redeem_loyalty_points", "cancel_loyalty_redemption"),
)

TOOL_WRITE_FAMILIES: Mapping[str, frozenset[str]] = {
    "add_to_cart": frozenset({"shopping.cart.item"}),
    "update_cart_item": frozenset({"shopping.cart.item"}),
    "remove_from_cart": frozenset({"shopping.cart.item"}),
    "apply_promo": frozenset({"shopping.cart.promo"}),
    "remove_promo": frozenset({"shopping.cart.promo"}),
    "redeem_loyalty_points": frozenset({"shopping.cart.loyalty_points"}),
    "cancel_loyalty_redemption": frozenset({"shopping.cart.loyalty_points"}),
    "set_shipping_option": frozenset({"shopping.cart.shipping_option"}),
}

REVERSIBLE_UNIT_FAMILIES = frozenset(
    {
        "shopping.cart.item",
        "shopping.cart.promo",
        "shopping.cart.loyalty_points",
        "shopping.cart.shipping_option",
        PRICE_REFRESH_FAMILY,
    }
)

# Every effect unit a write tool can contribute to a proposal footprint, over
# all argument and conversation contexts. ``update_cart_item`` additionally
# lists the price-refresh family because ``proposal_tool_write_families``
# reclassifies it in the stale-total context. Consumers may drop a tool schema
# only when this whole set misses the certified scope; tools absent from this
# mapping (reads) must always be kept.
CERTIFIED_PROJECTION_TOOL_UNITS: Mapping[str, frozenset[str]] = {
    name: (
        families | frozenset({PRICE_REFRESH_FAMILY})
        if name == "update_cart_item"
        else families
    )
    for name, families in TOOL_WRITE_FAMILIES.items()
}

REQUEST_FAMILY_TERMS: Mapping[str, tuple[str, ...]] = {
    "shopping.cart.item": (
        "cart",
        "add",
        "remove",
        "quantity",
        "gift wrap",
        "wrap",
        "total",
        "subtotal",
        "price",
        "cost",
    ),
    "shopping.cart.promo": ("promo", "coupon", "discount", "savings"),
    "shopping.cart.loyalty_points": ("loyalty", "points", "redeem"),
    "shopping.cart.shipping_option": ("shipping", "ship", "delivery"),
}

COMMITTED_REPAIR_FAMILY_TERMS: Mapping[str, tuple[str, ...]] = {
    "shopping.cart.item": (
        "item",
        "product",
        "quantity",
        "qty",
        "gift wrap",
        "wrap",
        "mug",
    ),
    "shopping.cart.promo": ("promo", "coupon", "discount", "savings"),
    "shopping.cart.loyalty_points": ("loyalty", "points", "redeem"),
    "shopping.cart.shipping_option": ("shipping", "ship", "delivery"),
}

_PROMO_WRITE_VERBS = ("apply", "use", "add", "remove", "delete", "redeem")
_PROMO_WRITE_MARKERS = ("promo", "coupon")
_PROMO_NONWRITE_RE = re.compile(
    r"\b(?:no|without)\s+(?:promo|promos|coupon|coupons)\b"
    r"|\b(?:leave|keep)\s+(?:promo|promos|coupon|coupons)\s+alone\b"
    r"|\b(?:do not|don't|dont)\s+(?:apply|use|add)\s+(?:a\s+)?(?:promo|coupon)\b"
)
_LOYALTY_WRITE_RE = re.compile(
    r"\b(?:redeem|use|apply|spend)\s+(?:the\s+)?(?:maximum\s+allowed\s+)?(?:loyalty\s+)?points\b"
    r"|\b(?:use|take)\s+the\s+(?:capped\s+)?points\s+path\b"
    r"|\b(?:cancel|remove|undo)\s+(?:the\s+)?(?:loyalty\s+)?(?:points|redemption)\b"
    r"|\bpoints?\s+(?:will\s+)?(?:come back|return|be returned|be refunded|"
    r"credited back)\b"
    r"|\bused\s+(?:loyalty\s+)?points\b.{0,160}\b(?:remove|removed|being removed)\b"
)
_LOYALTY_NONWRITE_RE = re.compile(
    r"\b(?:no|without)\s+(?:loyalty\s+)?points\b"
    r"|\b(?:leave|keep)\s+(?:loyalty\s+)?points\s+alone\b"
    r"|\b(?:leave|keep)\s+(?:promos?\s+and\s+)?points\s+alone\b"
    r"|\b(?:do not|don't|dont)\s+(?:redeem|use|apply|spend)\s+(?:loyalty\s+)?points\b"
)
_POINTS_DISCOUNT_INFO_RE = re.compile(
    r"\b(?:tell me|explain|what(?:'s| is)?|what happens|show)\b"
    r".{0,80}\b(?:points?|loyalty)\b.{0,40}\b(?:discount|redemption)\b"
)
_ITEM_WRITE_VERB_RE = re.compile(
    r"\b(add|remove|removed|removing|delete|update|change|set|increase|decrease|"
    r"adjust|replace|wrap|gift[- ]?wrap)\b"
)
_ITEM_NONWRITE_RE = re.compile(
    r"\b(?:do not|don't|dont)\s+add\s+(?:anything|it|one|a partial|partial)\b"
    r"|\bnot ready to add\b"
    r"|\bleave (?:the )?cart unchanged\b"
    r"|\bcart unchanged\b"
    r"|\bonly (?:want|wanted)\s+(?:the )?(?:stock|stock result|stock and size|comparison)\b"
    r"|\bjust tell me the exact stock\b"
    r"|\balready in (?:the )?cart\b"
    r"|\bnothing(?:'s| is| has been)?\s+(?:been\s+)?added to (?:the )?cart\b"
)
_CART_TOTAL_INFO_REQUEST_RE = re.compile(
    r"\b(?:tell me|what(?:'s| is)?|how much|current|currently|right now|latest|show)\b"
    r".{0,100}\b(?:cart\s+)?(?:total|subtotal|price|cost)\b"
    r"|\b(?:cart\s+)?(?:total|subtotal|price|cost)\b"
    r".{0,100}\b(?:tell me|what(?:'s| is)?|how much|current|currently|right now|latest|show)\b"
)
_DELIVERY_DEADLINE_RE = re.compile(
    r"\b(?:need|arrive|get|deliver|delivery)\b.{0,120}\bby\s+"
    r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|"
    r"\d{1,2}(?:st|nd|rd|th)?)\b"
    r"|\bby\s+"
    r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|"
    r"\d{1,2}(?:st|nd|rd|th)?)\b.{0,120}\b(?:arrive|deliver|delivery|get)\b"
)
_CART_SETUP_RE = re.compile(
    r"\b(?:add|put)\b.{0,80}\bcart\b|\bget me set up\b|\bset me up\b"
)
_SHIPPING_NONWRITE_RE = re.compile(
    r"\b("
    r"skip (?:it|shipping)"
    r"|(?:choose|pick|select|set)\s+shipping\s+at checkout"
    r"|handle (?:it|shipping) at checkout"
    r"|handle checkout later"
    r"|not (?:choosing|choose|selecting|select|picking|pick|setting|set) shipping"
    r"|(?:do not|don't|dont|no need to|do not need to|don't need to) "
    r"(?:pick|choose|select|set)?\s*shipping"
    r")\b"
)

# Visible markers of a *change/correction* request on an already-committed write.
# Deliberately conservative: this gates the ell2 repair frontier, so a missed
# repair (false negative) is preferred over opening a repair on an already-correct
# committed write (false positive), which would risk regressing a cheap-pass row.
#
# Strong markers imply an action on a prior commitment and are enough -- together
# with a reference token -- to attribute a pronoun-only change ("swap it instead")
# when exactly one reversible family is committed. Weak markers ("actually") are
# only honored when the sentence also names the family, to avoid false positives
# like "actually it looks great".
_STRONG_CHANGE_MARKERS = (
    "instead",
    "change",
    "changed my mind",
    "different",
    "replace",
    "swap",
    "switch",
    "no longer",
    "undo",
    "revert",
    "wrong",
    "mistake",
    "remove",
    "take off",
    "cancel that",
)
_PRONOUN_CHANGE_MARKERS = (
    "instead",
    "change",
    "changed my mind",
    "different",
    "replace",
    "swap",
    "switch",
    "no longer",
    "undo",
    "revert",
    "remove",
    "take off",
    "cancel that",
)
_WEAK_CHANGE_MARKERS = ("actually", "rather")
_COMMITTED_CHANGE_MARKERS = _STRONG_CHANGE_MARKERS + _WEAK_CHANGE_MARKERS
# Generic tokens that refer back to a prior commitment without naming its family.
_COMMITMENT_REFERENCE_TOKENS = frozenset(
    {"it", "that", "this", "them", "they", "one", "cart", "order"}
)
_COMMITTED_REPAIR_NEGATION_RE = re.compile(
    r"\b(?:do not|don't|dont|never|no need to|not)\s+"
    r"(?:\w+\s+){0,3}"
    r"(?:change|remove|delete|replace|swap|switch|undo|revert|take off|cancel)\b"
    r"|\b(?:not|nothing|isn't|isnt|aren't|arent)\s+"
    r"(?:\w+\s+){0,3}(?:wrong|mistake)\b"
    r"|\bno\s+(?:mistake|issue|problem)\b"
    r"|\b(?:keep|leave)\s+(?:it|that|this|them|the cart|cart)?\s*"
    r"(?:as is|unchanged)?\b"
    r"|\bas is\b"
)
_READ_ONLY_QUESTION_START_RE = re.compile(
    r"^\s*(?:is|are|was|were|do|does|did|can|could|would|should|what|how)\b"
)
_EXPLICIT_WRITE_REQUEST_RE = re.compile(
    r"\b(?:change|replace|swap|switch|remove|take off|undo|revert|set|update|"
    r"adjust|increase|decrease|use a different|different promo)\b"
)
_NON_REPAIR_CONFIRMATION_RE = re.compile(
    r"\b(?:looks great|looks good|is fine|okay|ok|no change|unchanged)\b"
)


def shopping_write_unit_signal(unit_family: str) -> WriteUnitSignal:
    """Return the schema-derived reversibility signal for a shopping unit."""

    reversible = unit_family in REVERSIBLE_UNIT_FAMILIES
    if unit_family == PRICE_REFRESH_FAMILY:
        reason = "shopping_cart_price_refresh_is_idempotent_in_place_recompute"
    else:
        reason = (
            "shopping_cart_surface_has_inverse_or_update_tool"
            if reversible
            else "shopping_unit_family_not_declared_reversible"
        )
    return WriteUnitSignal(
        unit_family=unit_family,
        in_place_update_tool=reversible,
        create_or_cancel_only=False,
        reason=reason,
    )


def restoration_cost_for_unit(unit: str) -> float:
    family = unit_family(unit)
    if family not in REVERSIBLE_UNIT_FAMILIES:
        return DEFAULT_RESTORATION_COST
    return restoration_cost_from_signal(shopping_write_unit_signal(family))


def restoration_cost_vector(units: Sequence[str]) -> dict[str, float]:
    """Build a schema-derived cost vector for shopping unit tokens."""

    return {unit: restoration_cost_for_unit(unit) for unit in units}


def unit_family(unit: str) -> str:
    return unit.split("#", 1)[0]


def state_write_contract_from_payload(payload: Mapping[str, Any]) -> StateWriteContract:
    """Build the visible shopping state-write contract for a cheap trajectory."""

    committed = _committed_units(payload)
    committed_families = frozenset(unit_family(unit) for unit in committed)
    residual_families = request_residual_families(payload)
    residual_witnesses = request_residual_family_quotes(payload)
    open_residual_families = residual_families - committed_families
    residual_units = frozenset(
        family
        for family in sorted(open_residual_families)
        if family in REVERSIBLE_UNIT_FAMILIES
    )
    closed = frozenset(
        Obligation(
            obligation_id=f"shopping.committed.{unit}",
            write_footprint=frozenset({unit}),
            postcondition_witness="visible_committed_shopping_write_tool",
            surface=SURFACE,
            status="closed",
        )
        for unit in sorted(committed)
    )
    residual = frozenset(
        Obligation(
            obligation_id=f"shopping.residual.{family}",
            write_footprint=frozenset({family}),
            postcondition_witness=residual_witnesses.get(
                family,
                "visible_user_request_or_schema_writable_family",
            ),
            surface=SURFACE,
            status="open_bounded",
        )
        for family in sorted(residual_units)
    )
    legal_actions = frozenset(
        {"bounded_repair", "abstain"} if residual else {"abstain"}
    )
    return StateWriteContract(
        surface=SURFACE,
        derivation_timing="runtime_progressive",
        closed=closed,
        residual=residual,
        forbidden_writes=committed,
        legal_actions=legal_actions,
        state_change_required=bool(residual),
        commitment_call_count=len(committed),
        commitment_footprint_size=len(committed),
        commitment_app_count=1 if committed else 0,
        soundness_assumptions=(
            "shopping_units_are_schema_level_cart_families_not_entity_specific",
            "shopping_reversibility_comes_from_inverse_or_update_tools",
        ),
        touched_units=committed | residual_units,
    )


def proposed_target_action_from_proposal(
    proposal: Any,
    payload: Mapping[str, Any] | None = None,
) -> ProposedTargetAction:
    """Project a residual proposal into an abstract shopping target action."""

    calls = _proposal_tool_calls(proposal)
    groups: list[frozenset[str]] = []
    names: list[str] = []
    for call in calls:
        name = str(call.get("name") or "")
        families = proposal_tool_write_families(call, payload)
        if not families:
            continue
        names.append(name)
        groups.append(families)
    write_footprint = frozenset(unit for group in groups for unit in group)
    action = _proposal_action(proposal)
    message = _proposal_assistant_message(proposal)
    return ProposedTargetAction(
        surface=SURFACE,
        action_id=f"shopping_residual:{action}:{','.join(names) or 'message_only'}",
        write_footprint=write_footprint,
        terminal_completion=bool(
            action == "execute" and message.strip() and not write_footprint
        ),
        effect_groups=tuple(groups),
    )


def request_residual_families(payload: Mapping[str, Any]) -> frozenset[str]:
    """Infer abstract writable shopping families requested in visible user text."""

    return frozenset(request_residual_family_quotes(payload))


def request_residual_family_quotes(payload: Mapping[str, Any]) -> dict[str, str]:
    """Map requested abstract write families to their first visible user quote."""

    witnesses: dict[str, str] = {}
    price_refresh_witness = price_refresh_residual_witness(payload)
    if price_refresh_witness:
        witnesses[PRICE_REFRESH_FAMILY] = price_refresh_witness
    shipping_deadline_witness = shipping_deadline_setup_witness(payload)
    if shipping_deadline_witness:
        witnesses["shopping.cart.shipping_option"] = shipping_deadline_witness
    for sentence in _visible_user_sentences(payload):
        lowered = sentence.lower()
        cancelled = _cancelled_write_families(lowered)
        for family in cancelled:
            witnesses.pop(family, None)
        for family, terms in REQUEST_FAMILY_TERMS.items():
            if family in witnesses or family in cancelled:
                continue
            if _sentence_requests_write_family(lowered, family, terms):
                witnesses[family] = sentence
    return witnesses


def shipping_deadline_setup_witness(payload: Mapping[str, Any]) -> str:
    for message in payload.get("conversation", []):
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        text = str(message.get("content") or "")
        lowered = text.lower()
        if _SHIPPING_NONWRITE_RE.search(lowered):
            continue
        if _DELIVERY_DEADLINE_RE.search(lowered) and _CART_SETUP_RE.search(lowered):
            return text
    return ""


def proposal_tool_write_families(
    call: Mapping[str, Any],
    payload: Mapping[str, Any] | None = None,
) -> frozenset[str]:
    """Contextual write footprint for proposal tool calls.

    ``update_cart_item`` is normally a cart-item write. In the narrow stale-total
    refresh context, the same tool with the same visible quantity/wrap arguments
    has the abstract effect of refreshing cached cart aggregates from live prices.
    """

    name = str(call.get("name") or "")
    if (
        name == "update_cart_item"
        and isinstance(payload, Mapping)
        and price_refresh_residual_witness(payload)
        and _update_cart_item_matches_price_refresh_target(call, payload)
    ):
        return frozenset({PRICE_REFRESH_FAMILY})
    return TOOL_WRITE_FAMILIES.get(name, frozenset())


def price_refresh_residual_witness(payload: Mapping[str, Any]) -> str:
    quote = _first_current_cart_total_quote(payload)
    if not quote:
        return ""
    if _read_probe_price_refresh_targets(payload) or _visible_price_refresh_targets(
        payload
    ):
        return quote
    return ""


def _update_cart_item_matches_price_refresh_target(
    call: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> bool:
    args = call.get("arguments") if isinstance(call.get("arguments"), Mapping) else {}
    if "quantity" not in args and "gift_wrap" not in args:
        return False
    targets = [
        *_read_probe_price_refresh_targets(payload),
        *_visible_price_refresh_targets(payload),
    ]
    for target in targets:
        if _update_args_match_price_refresh_target(args, target):
            return True
    return False


def _update_args_match_price_refresh_target(
    args: Mapping[str, Any],
    target: Mapping[str, Any],
) -> bool:
    customer_id = str(target.get("customer_id") or "").strip()
    product_id = str(target.get("product_id") or "").strip()
    if customer_id and str(args.get("customer_id") or "").strip() != customer_id:
        return False
    if product_id and str(args.get("product_id") or "").strip() != product_id:
        return False
    target_quantity = _money_number(target.get("quantity"))
    if "quantity" in args and target_quantity is not None:
        quantity = _money_number(args.get("quantity"))
        if quantity is None or abs(quantity - target_quantity) > 0.001:
            return False
    target_wrap = target.get("gift_wrap")
    if "gift_wrap" in args and target_wrap is not None:
        if bool(args.get("gift_wrap")) != bool(target_wrap):
            return False
    target_variant = str(target.get("variant_id") or "").strip()
    arg_variant = str(args.get("variant_id") or "").strip()
    if target_variant:
        return arg_variant == target_variant
    return not arg_variant


def _first_current_cart_total_quote(payload: Mapping[str, Any]) -> str:
    for sentence in _visible_user_sentences(payload):
        lowered = sentence.lower()
        if _CART_TOTAL_INFO_REQUEST_RE.search(lowered) and any(
            marker in lowered
            for marker in ("right now", "current", "currently", "now", "latest")
        ):
            return sentence
    return ""


def _read_probe_price_refresh_targets(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    context = payload.get(READ_BEFORE_MESSAGE_CONTEXT_KEY)
    if not isinstance(context, Mapping) or not context.get("enabled"):
        return ()
    families = {
        str(family)
        for family in context.get("open_dialogue_families", [])
        if isinstance(family, str)
    }
    if CURRENT_CART_TOTAL_DIALOGUE_FAMILY not in families:
        return ()
    carts = [
        row.get("result")
        for row in context.get("probe_results", [])
        if isinstance(row, Mapping)
        and str(row.get("family") or "") == "cart_current_snapshot"
        and isinstance(row.get("result"), Mapping)
    ]
    return tuple(
        target for cart in carts for target in _price_refresh_targets_from_cart(cart)
    )


def _visible_price_refresh_targets(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    targets: list[dict[str, Any]] = []
    saw_current_total_request = False
    for message in payload.get("conversation", []):
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role") or "")
        if role == "user":
            content = str(message.get("content") or "")
            if _CART_TOTAL_INFO_REQUEST_RE.search(content.lower()):
                saw_current_total_request = True
            continue
        if role != "assistant" or not saw_current_total_request:
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if (
                not isinstance(call, Mapping)
                or str(call.get("name") or "") != "get_cart"
            ):
                continue
            result = call.get("result")
            if isinstance(result, Mapping):
                targets.extend(_price_refresh_targets_from_cart(result))
    return tuple(targets)


def _price_refresh_targets_from_cart(
    cart: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    corrected_total = _corrected_cart_total(cart)
    visible_total = _money_number(cart.get("total")) or _money_number(
        cart.get("subtotal")
    )
    if corrected_total is None or visible_total is None:
        return ()
    if abs(visible_total - corrected_total) <= 0.001:
        return ()
    items = [item for item in cart.get("items", []) if isinstance(item, Mapping)]
    if len(items) != 1:
        return ()
    item = items[0]
    product_id = str(item.get("product_id") or "").strip()
    quantity = item.get("quantity")
    if not product_id or _money_number(quantity) is None:
        return ()
    return (
        {
            "customer_id": str(cart.get("customer_id") or "").strip(),
            "product_id": product_id,
            "variant_id": str(item.get("variant_id") or "").strip(),
            "quantity": quantity,
            "gift_wrap": item.get("gift_wrap"),
            "corrected_total": corrected_total,
            "visible_total": visible_total,
        },
    )


def _corrected_cart_total(cart: Mapping[str, Any]) -> float | None:
    line_total = 0.0
    saw_line = False
    for item in cart.get("items", []):
        if not isinstance(item, Mapping):
            continue
        item_line = _money_number(item.get("line_total"))
        if item_line is None:
            unit_price = _money_number(item.get("unit_price"))
            quantity = _money_number(item.get("quantity"))
            if unit_price is None or quantity is None:
                continue
            item_line = unit_price * quantity
        line_total += item_line
        saw_line = True
    if not saw_line:
        return None
    discount = _money_number(cart.get("discount_amount")) or 0.0
    loyalty_discount = _money_number(cart.get("loyalty_discount")) or 0.0
    gift_wrap_fee = _money_number(cart.get("gift_wrap_fee")) or 0.0
    shipping_cost = _money_number(cart.get("shipping_cost")) or 0.0
    return line_total - discount - loyalty_discount + gift_wrap_fee + shipping_cost


def _money_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip().replace("$", "").replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _sentence_requests_write_family(
    lowered_sentence: str,
    family: str,
    terms: tuple[str, ...],
) -> bool:
    if family == "shopping.cart.promo":
        return _sentence_requests_promo_write(lowered_sentence)
    if family == "shopping.cart.loyalty_points":
        return _sentence_requests_loyalty_points_write(lowered_sentence)
    if family == "shopping.cart.item":
        return _sentence_requests_item_write(lowered_sentence, terms)
    if family == "shopping.cart.shipping_option":
        return _sentence_requests_shipping_write(lowered_sentence, terms)
    return any(term in lowered_sentence for term in terms)


def _sentence_requests_item_write(
    lowered_sentence: str,
    terms: tuple[str, ...],
) -> bool:
    if _ITEM_NONWRITE_RE.search(lowered_sentence):
        return False
    if _CART_TOTAL_INFO_REQUEST_RE.search(
        lowered_sentence
    ) and not _ITEM_WRITE_VERB_RE.search(lowered_sentence):
        return False
    return _ITEM_WRITE_VERB_RE.search(lowered_sentence) is not None and any(
        term in lowered_sentence for term in terms
    )


def _sentence_requests_shipping_write(
    lowered_sentence: str,
    terms: tuple[str, ...],
) -> bool:
    if _SHIPPING_NONWRITE_RE.search(lowered_sentence):
        return False
    return any(term in lowered_sentence for term in terms)


def _sentence_requests_promo_write(lowered_sentence: str) -> bool:
    if _PROMO_NONWRITE_RE.search(lowered_sentence):
        return False
    if _POINTS_DISCOUNT_INFO_RE.search(lowered_sentence):
        return False
    if any(marker in lowered_sentence for marker in _PROMO_WRITE_MARKERS):
        words = frozenset(re.findall(r"[a-z]+", lowered_sentence))
        return bool(words & frozenset(_PROMO_WRITE_VERBS))
    if "discount" not in lowered_sentence and "savings" not in lowered_sentence:
        return False
    escaped_verbs = "|".join(re.escape(verb) for verb in _PROMO_WRITE_VERBS)
    return bool(
        re.search(
            rf"\b(?:{escaped_verbs})\b.{{0,40}}\b(?:discount|savings)\b"
            rf"|\b(?:discount|savings)\b.{{0,40}}\b(?:{escaped_verbs})\b",
            lowered_sentence,
        )
    )


def _sentence_requests_loyalty_points_write(lowered_sentence: str) -> bool:
    if _LOYALTY_NONWRITE_RE.search(lowered_sentence):
        return False
    if _POINTS_DISCOUNT_INFO_RE.search(lowered_sentence):
        return False
    return bool(_LOYALTY_WRITE_RE.search(lowered_sentence))


def _cancelled_write_families(lowered_sentence: str) -> frozenset[str]:
    cancelled: set[str] = set()
    if _ITEM_NONWRITE_RE.search(lowered_sentence):
        cancelled.add("shopping.cart.item")
    if _SHIPPING_NONWRITE_RE.search(lowered_sentence):
        cancelled.add("shopping.cart.shipping_option")
    if _PROMO_NONWRITE_RE.search(lowered_sentence):
        cancelled.add("shopping.cart.promo")
    if _LOYALTY_NONWRITE_RE.search(lowered_sentence):
        cancelled.add("shopping.cart.loyalty_points")
    return frozenset(cancelled)


def committed_change_repair_targets(payload: Mapping[str, Any]) -> frozenset[str]:
    """Committed, reversible cart families a user visibly asks to change post-commit.

    This is the scorer-blind trigger for the ell2 *temporal/irreversible repair*
    frontier. A family qualifies only when all three visible conditions hold:

    1. it was committed by a write tool call (so it is in ``forbidden_writes`` and
       excluded from the ordinary ell1 residual);
    2. it is schema-reversible (an inverse / in-place update tool exists), so the
       committed unit can be re-admitted within a restoration budget;
    3. a user turn *after* that commit requests a change to that family, using a
       conservative change/correction marker (not a mere later mention).

    Order matters: a change asked before the commit, or a later mention with no
    change marker, does not qualify. The function reads only the visible
    conversation, never scorer fields or task ids.
    """

    conversation = [
        message
        for message in payload.get("conversation", [])
        if isinstance(message, Mapping)
    ]
    committed_at: dict[str, int] = {}
    for index, message in enumerate(conversation):
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            if not _visible_write_call_succeeded(call):
                continue
            for family in TOOL_WRITE_FAMILIES.get(
                str(call.get("name") or ""), frozenset()
            ):
                if family in REVERSIBLE_UNIT_FAMILIES:
                    committed_at[family] = index
    if not committed_at:
        return frozenset()
    single_family = next(iter(committed_at)) if len(committed_at) == 1 else None

    targets: set[str] = set()
    for index, message in enumerate(conversation):
        if message.get("role") != "user":
            continue
        for sentence in request_cues.sentences(str(message.get("content") or "")):
            lowered = sentence.lower()
            if not _has_marker(lowered, _COMMITTED_CHANGE_MARKERS):
                continue
            has_strong = _has_marker(lowered, _STRONG_CHANGE_MARKERS)
            if _is_negated_or_read_only_repair_request(lowered, has_strong):
                continue
            named_families = _named_repair_families(lowered)
            has_pronoun_change = _has_marker(lowered, _PRONOUN_CHANGE_MARKERS)
            words = set(re.findall(r"[a-z']+", lowered))
            references_commitment = bool(words & _COMMITMENT_REFERENCE_TOKENS)
            for family, commit_index in committed_at.items():
                if family in targets or index <= commit_index:
                    continue
                names_family = family in named_families
                pronoun_change = (
                    single_family == family
                    and has_pronoun_change
                    and references_commitment
                    and not (named_families - {family})
                )
                if names_family or pronoun_change:
                    targets.add(family)
    return frozenset(targets)


def _has_marker(lowered_sentence: str, markers: Sequence[str]) -> bool:
    return any(_contains_phrase(lowered_sentence, marker) for marker in markers)


def _contains_phrase(lowered_sentence: str, phrase: str) -> bool:
    escaped = re.escape(phrase).replace(r"\ ", r"\s+")
    return bool(re.search(rf"(?<![a-z]){escaped}(?![a-z])", lowered_sentence))


def _is_negated_or_read_only_repair_request(
    lowered_sentence: str,
    has_strong_marker: bool,
) -> bool:
    if _COMMITTED_REPAIR_NEGATION_RE.search(lowered_sentence):
        return True
    if (
        "?" in lowered_sentence
        and _READ_ONLY_QUESTION_START_RE.search(lowered_sentence)
        and not _EXPLICIT_WRITE_REQUEST_RE.search(lowered_sentence)
    ):
        return True
    return not has_strong_marker and bool(
        _NON_REPAIR_CONFIRMATION_RE.search(lowered_sentence)
    )


def _mentions_family(lowered_sentence: str, family: str) -> bool:
    return any(
        term in lowered_sentence
        for term in COMMITTED_REPAIR_FAMILY_TERMS.get(family, ())
    )


def _named_repair_families(lowered_sentence: str) -> frozenset[str]:
    return frozenset(
        family
        for family in COMMITTED_REPAIR_FAMILY_TERMS
        if _mentions_family(lowered_sentence, family)
    )


def _committed_units(payload: Mapping[str, Any]) -> frozenset[str]:
    counts: Counter[str] = Counter()
    units: list[str] = []
    for call in residual_gates.extract_tool_calls_with_results(payload):
        if not _visible_write_call_succeeded(call):
            continue
        for family in TOOL_WRITE_FAMILIES.get(str(call.get("name") or ""), frozenset()):
            counts[family] += 1
            units.append(f"{family}#committed{counts[family]}")
    return frozenset(units)


def _visible_write_call_succeeded(call: Mapping[str, Any]) -> bool:
    name = str(call.get("name") or "")
    if name not in WRITE_TOOL_NAMES:
        return True
    result = call.get("result")
    if not isinstance(result, Mapping):
        return True
    if result.get("error"):
        return False
    status = str(result.get("status") or "").strip().lower()
    return status not in {"error", "failed", "rejected"}


def _visible_user_sentences(payload: Mapping[str, Any]) -> list[str]:
    sentences: list[str] = []
    for message in payload.get("conversation", []):
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        sentences.extend(request_cues.sentences(str(message.get("content") or "")))
    return sentences


def _proposal_tool_calls(proposal: Any) -> tuple[Mapping[str, Any], ...]:
    return residual_gates.proposal_tool_calls(proposal)


def _proposal_action(proposal: Any) -> str:
    if isinstance(proposal, Mapping):
        return str(proposal.get("action") or "")
    return str(getattr(proposal, "action", "") or "")


def _proposal_assistant_message(proposal: Any) -> str:
    if isinstance(proposal, Mapping):
        return str(proposal.get("assistant_message") or "")
    return str(getattr(proposal, "assistant_message", "") or "")


__all__ = [
    "SURFACE",
    "PRICE_REFRESH_FAMILY",
    "WRITE_TOOL_NAMES",
    "REVERSIBLE_TOOL_PAIRS",
    "TOOL_WRITE_FAMILIES",
    "CERTIFIED_PROJECTION_TOOL_UNITS",
    "REVERSIBLE_UNIT_FAMILIES",
    "shopping_write_unit_signal",
    "restoration_cost_for_unit",
    "restoration_cost_vector",
    "unit_family",
    "state_write_contract_from_payload",
    "proposed_target_action_from_proposal",
    "proposal_tool_write_families",
    "request_residual_families",
    "request_residual_family_quotes",
    "committed_change_repair_targets",
]

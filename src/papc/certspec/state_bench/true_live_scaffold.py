"""Slim STATE-Bench true-live scaffold and proposal compiler.

This module contains only the surface-specific helpers needed by the true-live
CFRC runner. It deliberately avoids importing the offline residual pilot base.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class ResidualGraphNode:
    """One model-authored action or terminal in an incremental residual graph."""

    node_id: str
    kind: Literal["tool_call", "terminal"]
    round_index: int
    dependency_ids: tuple[str, ...]
    carrier_digest: str
    receipt_digest: str = ""

    @classmethod
    def from_tool_call(
        cls,
        *,
        node_id: str,
        round_index: int,
        dependency_ids: Sequence[str],
        tool_call: Mapping[str, Any],
        receipt: Any,
    ) -> ResidualGraphNode:
        name = str(tool_call.get("name") or "")
        arguments = (
            dict(tool_call.get("arguments"))
            if isinstance(tool_call.get("arguments"), Mapping)
            else {}
        )
        return cls(
            node_id=node_id,
            kind="tool_call",
            round_index=int(round_index),
            dependency_ids=tuple(dependency_ids),
            carrier_digest=_stable_digest(
                {"name": name, "arguments": arguments}
            ),
            receipt_digest=_stable_digest(receipt),
        )

    @classmethod
    def terminal(
        cls,
        *,
        round_index: int,
        dependency_ids: Sequence[str],
        assistant_message: str,
    ) -> ResidualGraphNode:
        return cls(
            node_id="terminal",
            kind="terminal",
            round_index=int(round_index),
            dependency_ids=tuple(dependency_ids),
            carrier_digest=_stable_digest(str(assistant_message or "").strip()),
        )


@dataclass(frozen=True)
class IncrementalResidualGraph:
    """Receipt-grounded graph whose carriers are the compiled native calls."""

    nodes: tuple[ResidualGraphNode, ...]

    def validation_errors(
        self,
        *,
        assistant_message: str,
        tool_calls: Sequence[Mapping[str, Any]],
    ) -> tuple[str, ...]:
        errors: list[str] = []
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            errors.append("residual_graph_node_id_duplicate")
        tool_nodes = [node for node in self.nodes if node.kind == "tool_call"]
        terminal_nodes = [node for node in self.nodes if node.kind == "terminal"]
        if len(terminal_nodes) != 1 or not self.nodes or self.nodes[-1].kind != "terminal":
            errors.append("residual_graph_terminal_invalid")
        if len(tool_nodes) != len(tool_calls):
            errors.append("residual_graph_carrier_count_mismatch")
        if [node.carrier_digest for node in tool_nodes] != [
            _stable_digest(call) for call in tool_calls
        ]:
            errors.append("residual_graph_carrier_digest_mismatch")
        if [node.round_index for node in tool_nodes] != sorted(
            node.round_index for node in tool_nodes
        ):
            errors.append("residual_graph_round_order_invalid")

        seen_ids: set[str] = set()
        prior_tool_nodes: list[ResidualGraphNode] = []
        for node in self.nodes:
            if node.kind not in {"tool_call", "terminal"}:
                errors.append("residual_graph_node_kind_invalid")
            if any(dependency not in seen_ids for dependency in node.dependency_ids):
                errors.append("residual_graph_dependency_not_prior")
            if node.kind == "tool_call":
                expected_dependencies = tuple(
                    prior.node_id
                    for prior in prior_tool_nodes
                    if prior.round_index < node.round_index
                )
                if node.dependency_ids != expected_dependencies:
                    errors.append("residual_graph_receipt_dependency_mismatch")
                if not node.receipt_digest:
                    errors.append("residual_graph_receipt_missing")
                prior_tool_nodes.append(node)
            seen_ids.add(node.node_id)

        if terminal_nodes:
            terminal = terminal_nodes[0]
            expected_dependencies = tuple(node.node_id for node in tool_nodes)
            if terminal.dependency_ids != expected_dependencies:
                errors.append("residual_graph_terminal_dependency_mismatch")
            if tool_nodes and terminal.round_index <= tool_nodes[-1].round_index:
                errors.append("residual_graph_terminal_round_invalid")
            if terminal.carrier_digest != _stable_digest(
                str(assistant_message or "").strip()
            ):
                errors.append("residual_graph_terminal_digest_mismatch")
        return tuple(sorted(set(errors)))

    def public_summary(self) -> dict[str, Any]:
        tool_nodes = tuple(node for node in self.nodes if node.kind == "tool_call")
        terminal_nodes = tuple(node for node in self.nodes if node.kind == "terminal")
        return {
            "schema": "papc.incremental_residual_graph.v1",
            "node_count": len(self.nodes),
            "tool_node_count": len(tool_nodes),
            "edge_count": sum(len(node.dependency_ids) for node in self.nodes),
            "round_count": len({node.round_index for node in tool_nodes}),
            "terminal_node_present": len(terminal_nodes) == 1,
            "receipt_grounded_tool_node_count": sum(
                1 for node in tool_nodes if node.receipt_digest
            ),
            "carrier_payload_publicly_disclosed": False,
            "receipt_payload_publicly_disclosed": False,
        }


@dataclass(frozen=True)
class CompiledTrueLiveProposal:
    decision: Literal["execute", "abstain", "block"]
    assistant_message: str = ""
    tool_calls: tuple[dict[str, Any], ...] = ()
    compile_errors: tuple[str, ...] = ()
    residual_graph: IncrementalResidualGraph | None = None

    def public_summary(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "assistant_message_present": bool(self.assistant_message.strip()),
            "tool_call_count": len(self.tool_calls),
            "compile_errors": list(self.compile_errors),
            "residual_graph": (
                self.residual_graph.public_summary()
                if self.residual_graph is not None
                else None
            ),
        }


def load_state_bench(root: Path) -> dict[str, Any]:
    """Import the official STATE-Bench package from ``root``."""

    root = root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from state_bench.domain import get_domain_config
    from state_bench.env_loader import load_task_environment
    from state_bench.paths import domain_tasks_dir
    from state_bench.protocol import load_default_protocol, load_split_task_ids
    from state_bench.schemas import StateDiff, TaskDefinition

    protocol = load_default_protocol()
    return {
        "load_task_environment": load_task_environment,
        "domain_tasks_dir": domain_tasks_dir,
        "get_domain_config": get_domain_config,
        "load_split_task_ids": load_split_task_ids,
        "protocol_split_version": protocol.split_version,
        "StateDiff": StateDiff,
        "TaskDefinition": TaskDefinition,
    }


DOMAIN_DISCIPLINE_MODES = ("full", "generic", "none")

# The value-free form of the second section. The full form names thresholds
# ("120 minutes", "5+ cart items", "3-5 business days"), tools, and tool
# arguments ("call update_cart_item on the same product_id, quantity, and
# gift_wrap"). Every one of those values lives in the benchmark's own
# domains/*/policies.py as server-side logic that the agent is expected to
# reach through a read tool: the published agent prompt is 39 tokens of role
# framing and says to use the tools to inspect policies. Supplying them in a
# system prompt hands the agent the result of a read it never performed, which
# AGENTS.md forbids as a fixed tool argument and a benchmark-specific answer
# sketch. This form keeps the obligation - close the policy loop, disclose the
# arithmetic, confirm before an item-level write, do not promise what the
# policy defers - and drops every value, tool name, and scenario recipe, so
# the discipline transfers to any domain with tiered policies rather than
# encoding this benchmark's answers.
_GENERIC_DOMAIN_DISCIPLINE: dict[str, str] = {
    "customer_support": """# Customer Support Closure Discipline
When a residual action changes order, return, refund, cancellation, warranty,
exchange, shipping-claim, or compensation state, the final user-facing message
must close the visible policy loop: state the policy basis, the amounts, the
timelines, the eligibility condition, and any clawback or deferred-resolution
constraint that the tool results established.

Before item-level writes for a multi-item request, confirm the exact items and
scope unless the live prefix already confirmed that enumerated scope. After
execution, enumerate the affected items and their fee impact.

Quote a visible policy term with the value the tool returned rather than
paraphrasing it into a rounded or generic form. When a promotion affects an
amount, name the visible promotion and describe the allocation the tool result
reflects.

Do not promise an immediate outcome when the visible policy defers it to
review, investigation, or claim completion. When the user's reference is
ambiguous or the case admits several remedy paths, ask before writing. Do not
offer a remedy, exception, or escalation path that no visible policy result
supports.""",
    "travel": """# Travel Policy and Booking Closure Discipline
When a residual action changes or explains a booking, cancellation, delay,
refund, hotel, add-on, baggage, or compensation outcome, the final message must
make the governing threshold and the tradeoff explicit, using the values the
tools returned.

Where a policy is tiered, name the tier that applied and, when it is visible,
the adjacent tier the case did not reach. Where a fee depends on timing, state
the applicable window and whether waiting would move the booking into a more
expensive tier. Preserve the visible reason category and state the exact fee
the tool applied.

For a search, name the preferred option when it is visible and say which
constraint it failed. When choosing an alternative, say which preference it
keeps and which it trades off; never substitute one silently.

Where more than one path exists, compare them before recommending, covering
each path's fees, differences, refunds, and net cost wherever those are
visible. Give the components and the total exactly as the tool results support
and never round an exact tool amount. Where an allowance is composed of a base
and a bonus, give that arithmetic explicitly.

For a cleanup, cancellation, or rebooking request, look for visible linked
components rather than treating the request as a single item, and warn about
what a partial action would orphan before taking it.""",
    "shopping_assistant": """# Shopping Cart Completion Discipline
When the live conversation concerns cart readiness, checkout, discounts,
promotions, fees, totals, or loyalty points, every current-cart or final
confirmation must include the complete visible numeric breakdown the user
needs to verify the answer: the subtotal, each discount and fee that is
visible or applied, the final total, and loyalty points when asked or visible.

If any of those values may have changed after a tool call, read the current
state before answering. If the latest user message is a short closing or
approval after updates, treat your reply as the final confirmation and include
that breakdown again.

Never probe with a mutating tool. When a visible read shows the requested write
needs clamping, substitution, or a smaller quantity, explain that constraint
and wait for explicit confirmation instead of writing and repairing. Surface a
budget overage and let the user choose before committing the write.

Never state that a mutating call was executed unless it appears in the same
assistant turn. When the user's decision requires a write for the visible state
to match your description, perform that write before the final confirmation
rather than describing the outcome in text.

Trust the authoritative visible tool result over a user's claimed value: do not
proactively search for discounts, price matches, or courtesy adjustments unless
the user asks, and do not concede that the result is stale without a fresh tool
result showing it. When an aggregate conflicts with the item-level values it
should summarise, re-read the affected item rather than trusting the aggregate,
and mention any change you find.""",
}


def live_online_system_prompt_suffix(
    domain_name: str,
    *,
    discipline: str = "full",
    include_domain_discipline: bool | None = None,
) -> str:
    """Return the live online continuation instructions for one domain.

    The first section is domain independent and states the method itself: the
    prefix is a verifier-accepted cheap prefix, the saved suffix is not
    available, already executed commitments are preserved, and missing evidence
    is read rather than invented.

    The second section states the closure discipline for the domain, in one of
    three forms. ``full`` is the incumbent text, which names this benchmark's
    thresholds, tools, and tool arguments. ``generic`` states the same
    obligations with every value, tool name, and scenario recipe removed.
    ``none`` drops the section, which is the arm whose prompt is matched to the
    cheap anchor while keeping the method intact.
    """

    if include_domain_discipline is not None:
        discipline = "full" if include_domain_discipline else "none"
    if discipline not in DOMAIN_DISCIPLINE_MODES:
        raise ValueError(
            f"discipline must be one of {DOMAIN_DISCIPLINE_MODES}: {discipline!r}"
        )

    common = """# PAPC Live Online Frontier
You are continuing a live STATE-Bench episode after a verifier-accepted cheap
prefix. The transcript you see is the complete live transcript so far. You do
not have access to the cheap agent's saved future suffix, and you must not infer
one. Use only the current conversation, allowed tool schemas, and tool results
returned during this live run.

Preserve already executed commitments from the visible prefix. When a user asks
for a follow-up, complete the smallest legal remaining action or answer. If
current visible evidence is insufficient, call an allowed read tool instead of
inventing IDs, prices, dates, policies, fees, totals, or statuses."""
    if discipline == "none":
        return common
    if discipline == "generic":
        section = _GENERIC_DOMAIN_DISCIPLINE.get(domain_name)
        return common if section is None else common + "\n\n" + section
    if domain_name == "customer_support":
        customer = """# Customer Support Closure Discipline
When a residual action changes order, return, refund, cancellation, warranty,
exchange, shipping-claim, or compensation state, the final user-facing message
must close the visible policy loop. State the exact policy basis, amounts,
timelines, eligibility condition, and any clawback or non-immediate-resolution
constraint that was used by the tool results.

For a multi-item return, exchange, cancellation, or warranty action, first get
explicit confirmation of the exact items and scope before executing item-level
writes unless the user has already confirmed that enumerated scope in the live
prefix. After execution, enumerate the affected items and refund/fee impact.

When a promotion affects a refund, name the visible promo code and explain that
the discount was proportionally allocated to the returned item if the tool or
policy result reflects allocation. For high-value lost or missing shipments,
include the exact investigation timeline when visible, including 3-5 business days.
For refurbished warranty cases, use the exact visible warranty term such
as 90-day refurbished warranty instead of paraphrasing it as a generic month
count. Do not promise immediate refund, reshipment, replacement, or compensation
when the visible policy requires review, investigation, or claim completion."""
        customer += """

Item-level write with a multi-item request, a challenge, an ambiguous item reference, or a choice among refund, reship, exchange, store credit, compensation destination, or denial/escalation paths in the live prefix: send a message-only preview, wait for the user's explicit confirmation, and keep the mutating write out of that same assistant turn. Ambiguous item reference: ask a clarifying question before any return, refund, exchange, cancellation, or credit tool call.

Promise no immediate refund, reshipment, replacement, or compensation while the visible policy requires review, investigation, or claim completion. Delivered-not-received claim with a signature on file: deny the immediate refund or replacement on that signed-delivery basis, and offer no investigation, exception, courtesy replacement, or escalation path unless a visible policy result explicitly requires that path. After denying an expired changed-mind return, offer no defective, exception, resale, escalation, or future flexibility path."""
        return common + "\n\n" + customer
    if domain_name == "travel":
        travel = """# Travel Policy and Booking Closure Discipline
When a residual action changes or explains a booking, cancellation, delay,
refund, hotel, add-on, baggage, or compensation outcome, the final message must
make the policy threshold and tradeoff explicit. Include the exact visible fee,
refund, delay threshold, time window, cabin/add-on choice, and exception reason
that determined the action.

For delay compensation, state both the lower eligibility threshold such as
120 minutes and any higher full-compensation threshold such as 240 minutes when
the visible policy contains those tiers. For hotel cancellation, state the
applicable window such as 24-48 hour, the fee tier, and whether waiting would
move the booking into a more expensive tier. For flight changes, preserve the
visible reason category such as medical, airline-caused, or personal and state
the exact discounted or waived fee that the tool result applied.

For booking searches, name the preferred itinerary or option when visible and
explain why it failed a constraint such as budget, timing, cabin, stops, or
airline. When choosing an alternative, explicitly say which preference it keeps
and which preference it trades off. Do not silently book an alternative without
the visible tradeoff that justified it."""
        travel += """

Strategy, cancellation, rebooking, or change task with multiple paths: compare both paths before recommending or executing, and recommend only afterwards, covering direct-change fee and fare difference against cancellation fee, refund amount, replacement fare, and net out-of-pocket wherever visible.

Every travel fee explanation must give the base fee, timing tier, reason discount, fare difference, and final total exactly as the visible tool or policy result supports. Cheaper replacement fare: state whether the negative fare difference is not refunded and what that means for the updated total against the original booking. Never round exact tool amounts or quote half-dollar estimates against an integer tool fee. Points or budget task: explain remaining points, cash cap, and why an alternative over the cap is invalid before booking."""
        travel += """

Trip cleanup, cancellation, refund, or rebooking request: look proactively for visible linked travel components instead of treating the request as a single flight; mention any same-day onward connection in the reservations and warn about orphaning the remaining leg before the first-leg cancellation; surface any destination hotel reservation and car rental visible or discoverable through allowed reservation reads before calling the trip cleanup complete.

Every baggage and loyalty explanation must give the per-booking allowance with explicit arithmetic: cabin base allowance plus the platinum bonus or other visible loyalty bonus equals the final free checked-bag count. For a budgeted booking alternative, name the over-budget preferred option and the within-budget tradeoff before booking."""
        return common + "\n\n" + travel
    if domain_name != "shopping_assistant":
        return common
    shopping = """# Shopping Cart Completion Discipline
When the live conversation concerns cart readiness, checkout, discounts,
promotions, gift wrap, shipping, fees, totals, or loyalty points, every
current-cart or final confirmation must include the complete visible numeric
breakdown needed for the user to verify the answer: subtotal, promo discount amount
when a promotion is visible or applied, gift wrap fee when gift wrap is visible
or applied, shipping or other fees when visible, final total, and loyalty points
earned or redeemed when asked or visible.

If any one of those values may have changed after a tool call, read the current
cart/account/policy before answering. If the latest user message is a short
closing or approval such as thanks, perfect, ok, done, or sounds good after
cart updates or points discussion, treat your reply as the final confirmation
and include that complete visible numeric breakdown again. Do not use task identifiers,
hidden gold, scorer outcomes, private fixed handles, or answer sketches."""
    shopping += """

Never probe with mutating cart tools. Visible cart read, product read, policy, fee, or budget constraint showing the requested write needs clamping, substitution, skipping an option, or a smaller quantity: explain that constraint and wait for explicit user confirmation first. With 3 of a product already in the visible cart and a request to add another of it, read the quantity_limit policy when available, explain the per-product cap, and ask for or suggest a different item rather than probing with add_to_cart.

New fee or option over a visible budget: surface the budget overage and let the user choose before committing the write; never add the over-budget option first and repair it after the user objects.

User asked explicitly for free shipping, visible shipping policy grants free standard shipping at 5+ cart items, and the dialogue reached or confirmed the needed extra item: set the standard shipping option before the final cart confirmation rather than stopping at a message-only "shipping is free" explanation while shipping_option is unset; a follow-up like "is shipping free now?" is that confirmation. Never claim in text that a mutating tool call was executed unless it appears in the same assistant turn. Shipping option chosen while shipping_option is unset: call set_shipping_option before final confirmation; for a newly added item, add it first, then set the chosen option against the updated cart.

User challenge to an authoritative tool price: trust the visible tool result; never proactively search for promotions, discounts, price matches, manual adjustments, or courtesy bridges unless the user explicitly asks for those options; never concede that the tool/database may be stale without a fresh visible tool result showing it is stale; never offer to check or use loyalty points, promotions, discounts, price-match policies, or account credits as a bridge to the user's claimed lower price. If the user asks to use a lower claimed price, state that you can transact only at the current tool price unless a fresh visible tool result changes it.

Cart subtotal/total conflicting with item unit_price or line_total: never trust it silently; call get_product_details on the affected product_id when available, compare price against previous_price, and mention any price drop. Without discounts, fees, gift wrap, shipping, or loyalty redemption, the charged total should follow the current item line_total / current product price, not the stale cart subtotal/total.

If the user proceeds toward checkout after that stale cart-total conflict, perform only a bounded cart-item refresh before the final checkout-ready confirmation: call update_cart_item on the same product_id, quantity, and gift_wrap already in the cart, changing no product, quantity, shipping, promo, or loyalty state."""
    return common + "\n\n" + shopping


def strong_json_proposal_instruction(
    *,
    system_prompt: str,
    strong_system_prompt_suffix: str,
    frontier: Mapping[str, Any] | None,
) -> str:
    frontier_json = json.dumps(
        frontier or {},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    suffix = str(strong_system_prompt_suffix or "").strip()
    return "\n\n".join(
        part
        for part in (
            system_prompt.strip(),
            suffix,
            "# CFRC Residual Proposal Contract\n"
            "You are not taking over the live agent. Return exactly one JSON object "
            "and no surrounding prose. The runner will compile it before any "
            "message or tool call reaches the harness.\n\n"
            "Allowed shape:\n"
            "{\"action\":\"execute\",\"assistant_message\":\"...\",\"tool_calls\":[{\"name\":\"tool_name\",\"arguments\":{}}]}\n"
            "or {\"action\":\"abstain\",\"assistant_message\":\"short reason\",\"tool_calls\":[]}.\n"
            "Use only visible conversation evidence and allowed tool schemas. "
            "Do not include task ids, gold outcomes, private paths, or raw scorer data.\n\n"
            f"Frontier summary: {frontier_json}",
        )
        if part
    )


@dataclass(frozen=True)
class SuccessorContractPromptView:
    """Render the contract-related facts shown in the successor prompt.

    This view is not a serialized ``z_f`` and its fields are not the formal
    ``K``, ``E``, ``O``, ``S``, and ``A`` components. It only assembles
    prompt-visible transport and runtime facts: where the frontier summary is
    carried, the compiler's tool-call ceiling, the remaining response
    allowance, and an optional description of terminal-message admission.
    Contract construction and exact residual authority remain in the reached
    state and deterministic compiler/checker.

    Keep rendering byte-stable unless an experiment deliberately changes the
    successor instruction; renaming this internal view must not reprice an arm.
    """

    frontier: Mapping[str, Any] | None = None
    frontier_in_input_packet: bool = False
    max_tool_calls: int | None = None
    remaining_responses: int | None = None
    declare_closure_admissibility: bool = False
    closing_contract_style: str = "consequence"

    def frontier_section(self) -> str:
        if self.frontier_in_input_packet:
            return "Frontier summary: supplied in the structured residual context."
        frontier_json = json.dumps(
            self.frontier or {},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"Frontier summary: {frontier_json}"

    def tool_call_budget_section(self) -> str | None:
        if self.max_tool_calls is None:
            return None
        budget = max(0, int(self.max_tool_calls))
        return (
            f"Admissible action budget: the compiled residual may carry at most "
            f"{budget} tool calls in total, counting every call you make while "
            "working here. A proposal that passes the budget is rejected whole, "
            "so nothing in it runs. Keep exploration inside the budget. If the "
            "residual cannot be completed within it, do not author a partial "
            "action set: return an assistant message that says what is missing "
            "and hands control back."
        )

    def response_budget_section(self) -> str | None:
        if self.remaining_responses is None:
            return None
        return (
            "Residual response budget: "
            f"{max(0, int(self.remaining_responses))} successor responses remain "
            "in this contract tenure, including this response and any closing "
            "rewrite. Use this response to advance or close the admissible "
            "residual; do not assume another response will be available."
        )

    def closing_contract_section(self) -> str:
        """Tell the proposer to close in the round that acts.

        Two wordings have been run. ``plain`` states the requirement and was the
        wording in place when this pool measured 94 of 150. ``consequence`` adds
        what the compiler does to a response that acts without text, which was
        added after 450 rows in which the proposer never once closed in the
        round that acted. Which one a run uses is recorded, because they are
        different instructions to the same model.
        """

        head = (
            "Closing contract: return the closing message for the user in the same "
            "response that carries your actions"
        )
        if self.closing_contract_style == "plain":
            return (
                head + ". State what those actions did and "
                "what it means for the user, in the values they will see, not as a plan "
                "for a later turn. When nothing admissible is left to add, return the "
                "message with no tool call."
            )
        return (
            head + ", as assistant text beside the tool "
            "calls rather than in a later turn. State what those actions did and "
            "what it means for the user, in the values they will see, not as a plan "
            "for a later turn. When nothing admissible is left to add, return the "
            "message with no tool call. A response that carries actions and no text "
            "cannot be closed: the runner discards that proposal whole and none of "
            "the actions in it run, so the round is spent and the residual is lost. "
            "Never answer with tool calls alone."
        )

    def closure_admissibility_section(self) -> str | None:
        if not self.declare_closure_admissibility:
            return None
        return (
            "Closure admissibility: before your closing message reaches the "
            "user, the runner checks it against the receipts your actions "
            "returned. The message is admissible only when every value it "
            "states about what you changed matches the receipt for that "
            "change, when it states no value that neither the request nor a "
            "tool result you received carries, and when any set the request "
            "ranges over has been read so its members can be named rather than "
            "summarized. A message that fails this check is rewritten before it "
            "is sent, so satisfy the conditions in the message you first return."
        )


def strong_native_proposal_instruction(
    *,
    system_prompt: str,
    strong_system_prompt_suffix: str,
    frontier: Mapping[str, Any] | None,
    frontier_in_input_packet: bool = False,
    max_tool_calls: int | None = None,
    remaining_residual_rounds: int | None = None,
    closure_obligation_declared: bool = False,
    closing_contract_style: str = "consequence",
) -> str:
    prompt_view = SuccessorContractPromptView(
        frontier=frontier,
        frontier_in_input_packet=frontier_in_input_packet,
        max_tool_calls=max_tool_calls,
        remaining_responses=remaining_residual_rounds,
        declare_closure_admissibility=closure_obligation_declared,
        closing_contract_style=closing_contract_style,
    )
    suffix = str(strong_system_prompt_suffix or "").strip()
    frontier_location = prompt_view.frontier_section()
    action_budget = prompt_view.tool_call_budget_section()
    residual_round_budget = prompt_view.response_budget_section()
    # The message the proposer returns alongside its actions is the one the user
    # receives once those actions close the residual. Left unstated, the proposer
    # treats every round as preparatory and writes the closing message only after
    # being asked again, which costs one further call for text it could have
    # returned with the actions it already chose.
    # This instruction describes existing compiler behavior; it does not define
    # the request-scoped open-obligation set in the reached-state contract.
    completion_contract = prompt_view.closing_contract_section()
    closure_obligations = prompt_view.closure_admissibility_section()
    return "\n\n".join(
        part
        for part in (
            system_prompt.strip(),
            suffix,
            "# CFRC Residual Proposal Contract\n"
            "You are not taking over the live agent from scratch. You are proposing "
            "the next residual-completion assistant turn from the current visible "
            "prefix. The runner will compile your message and tool calls before "
            "anything reaches the harness.\n\n"
            "Use the native tool interface when a tool is needed. Use only visible "
            "conversation evidence, allowed tool schemas, and tool results returned "
            "during this live run. Do not include task ids, gold outcomes, private "
            "paths, raw scorer data, or hidden answer sketches.",
            action_budget,
            residual_round_budget,
            completion_contract,
            closure_obligations,
            frontier_location,
        )
        if part
    )


def compile_true_live_proposal(
    raw_text: str,
    *,
    tools: Sequence[Mapping[str, Any]],
    max_tool_calls: int = 8,
) -> CompiledTrueLiveProposal:
    """Compile a strong JSON proposal into a harness-safe assistant turn."""

    raw = _parse_json_object(raw_text)
    if raw is None:
        return CompiledTrueLiveProposal(
            decision="block",
            compile_errors=("proposal_json_invalid",),
        )
    action = str(raw.get("action") or "").strip().lower()
    if action == "abstain":
        return CompiledTrueLiveProposal(
            decision="abstain",
            assistant_message=str(raw.get("assistant_message") or "").strip(),
            compile_errors=(),
        )
    if action != "execute":
        return CompiledTrueLiveProposal(
            decision="block",
            compile_errors=("proposal_action_invalid",),
        )

    assistant_message = str(raw.get("assistant_message") or "").strip()
    tool_calls_value = raw.get("tool_calls")
    tool_calls = _normalise_tool_calls(tool_calls_value)
    if tool_calls is None:
        return CompiledTrueLiveProposal(
            decision="block",
            compile_errors=("proposal_tool_calls_invalid",),
        )
    return compile_native_true_live_proposal(
        assistant_message=assistant_message,
        tool_calls=tool_calls,
        tools=tools,
        max_tool_calls=max_tool_calls,
    )


def compile_native_true_live_proposal(
    *,
    assistant_message: str,
    tool_calls: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    max_tool_calls: int = 8,
    residual_graph: IncrementalResidualGraph | None = None,
) -> CompiledTrueLiveProposal:
    """Compile a native tool-call proposal into a harness-safe assistant turn."""

    normalised = _normalise_tool_calls(tool_calls)
    errors: list[str] = []
    if normalised is None:
        errors.append("proposal_tool_calls_invalid")
        normalised = ()
    if len(normalised or ()) > max(0, int(max_tool_calls)):
        errors.append("proposal_tool_call_budget_exceeded")
    allowed = _allowed_tool_names(tools)
    if allowed:
        for call in normalised or ():
            if str(call.get("name") or "") not in allowed:
                errors.append("proposal_tool_name_not_allowed")
                break
    elif normalised:
        errors.append("proposal_tool_name_not_allowed")
    assistant_message = str(assistant_message or "").strip()
    if not assistant_message and not normalised:
        errors.append("proposal_empty")
    if residual_graph is not None:
        errors.extend(
            residual_graph.validation_errors(
                assistant_message=assistant_message,
                tool_calls=normalised or (),
            )
        )
    if errors:
        return CompiledTrueLiveProposal(
            decision="block",
            compile_errors=tuple(sorted(set(errors))),
            residual_graph=residual_graph,
        )
    return CompiledTrueLiveProposal(
        decision="execute",
        assistant_message=assistant_message,
        tool_calls=tuple(normalised or ()),
        compile_errors=(),
        residual_graph=residual_graph,
    )


def apply_true_live_precommit_guard(
    *,
    domain: str,
    conversation: Sequence[Mapping[str, Any]],
    compiled: CompiledTrueLiveProposal,
    strong_calls_remaining: int,
) -> CompiledTrueLiveProposal:
    """Force high-risk writes through a message-only confirmation turn."""

    if compiled.decision != "execute" or not compiled.tool_calls:
        return compiled
    if int(strong_calls_remaining) <= 0:
        return compiled
    mutating_names = _precommit_mutating_tool_names(domain)
    if not mutating_names:
        return compiled
    if not any(str(call.get("name") or "") in mutating_names for call in compiled.tool_calls):
        return compiled
    if _latest_user_message_confirms(conversation):
        return compiled
    return CompiledTrueLiveProposal(
        decision="execute",
        assistant_message=_precommit_confirmation_message(domain),
        tool_calls=(),
        compile_errors=tuple(
            sorted({*compiled.compile_errors, "precommit_confirmation_required"})
        ),
    )


def true_live_precommit_requires_confirmation(
    *,
    domain: str,
    conversation: Sequence[Mapping[str, Any]],
    compiled: CompiledTrueLiveProposal,
) -> bool:
    """Return whether a compiled mutating suffix should be staged for confirmation."""

    if compiled.decision != "execute" or not compiled.tool_calls:
        return False
    if not true_live_has_mutating_precommit_tool(
        domain=domain,
        tool_calls=compiled.tool_calls,
    ):
        return False
    return not latest_user_confirms_precommit(conversation)


def true_live_has_mutating_precommit_tool(
    *,
    domain: str,
    tool_calls: Sequence[Mapping[str, Any]],
) -> bool:
    mutating_names = _precommit_mutating_tool_names(domain)
    if not mutating_names:
        return False
    return any(str(call.get("name") or "") in mutating_names for call in tool_calls)


def latest_user_confirms_precommit(conversation: Sequence[Mapping[str, Any]]) -> bool:
    return _latest_user_message_confirms(conversation)


def staged_true_live_precommit_message(
    *,
    domain: str,
    compiled: CompiledTrueLiveProposal,
) -> str:
    """Build a message-only preview for a cached mutating suffix."""

    draft = str(compiled.assistant_message or "").strip()
    if not draft:
        return _precommit_confirmation_message(domain)
    return (
        "Before I execute this change, please confirm the planned action. "
        f"{draft}"
    )


def _precommit_mutating_tool_names(domain: str) -> frozenset[str]:
    if domain == "customer_support":
        return frozenset(
            {
                "cancel_order",
                "process_exchange",
                "process_refund",
                "process_return",
            }
        )
    if domain == "travel":
        return frozenset(
            {
                "cancel_booking",
                "cancel_car_rental",
                "cancel_hotel_reservation",
                "create_booking",
                "create_car_rental",
                "create_hotel_reservation",
                "update_booking",
            }
        )
    return frozenset()


def _latest_user_message_confirms(conversation: Sequence[Mapping[str, Any]]) -> bool:
    for message in reversed(conversation):
        if message.get("role") != "user":
            continue
        text = str(message.get("content") or "").strip().lower()
        if "[task_done]" in text:
            return False
        if re.search(r"\b(no|stop|hold|wait|do\s+not|don't|dont)\b", text):
            return False
        words = set(re.findall(r"[a-z]+", text))
        if words & {
            "yes",
            "okay",
            "ok",
            "confirm",
            "confirmed",
            "approve",
            "approved",
            "proceed",
            "finalize",
            "finalise",
        }:
            return True
        confirmation_phrases = (
            "go ahead",
            "do it",
            "book it",
            "cancel it",
            "process it",
            "sounds good",
        )
        return any(phrase in text for phrase in confirmation_phrases)
    return False


def _precommit_confirmation_message(domain: str) -> str:
    if domain == "travel":
        return (
            "Before I execute any booking, cancellation, or change, please confirm "
            "the proposed option or strategy. I have not made the tool change yet; "
            "once you confirm, I will proceed with the selected action."
        )
    return (
        "Before I execute this customer-support change, please confirm the exact "
        "scope and resolution you want. I have not made the tool change yet; once "
        "you confirm, I will proceed with the selected action."
    )


def _parse_json_object(raw_text: str) -> dict[str, Any] | None:
    text = str(raw_text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _normalise_tool_calls(value: Any) -> tuple[dict[str, Any], ...] | None:
    if value in (None, ""):
        return ()
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return None
    calls: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            return None
        name = str(item.get("name") or "").strip()
        arguments = item.get("arguments")
        if not name or not isinstance(arguments, Mapping):
            return None
        calls.append({"name": name, "arguments": dict(arguments)})
    return tuple(calls)


def _allowed_tool_names(tools: Sequence[Mapping[str, Any]]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        direct = str(tool.get("name") or "").strip()
        if direct:
            names.add(direct)
        function = tool.get("function")
        if isinstance(function, Mapping):
            function_name = str(function.get("name") or "").strip()
            if function_name:
                names.add(function_name)
    return names


def _stable_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "CompiledTrueLiveProposal",
    "IncrementalResidualGraph",
    "ResidualGraphNode",
    "apply_true_live_precommit_guard",
    "compile_native_true_live_proposal",
    "compile_true_live_proposal",
    "live_online_system_prompt_suffix",
    "load_state_bench",
    "strong_json_proposal_instruction",
    "strong_native_proposal_instruction",
]

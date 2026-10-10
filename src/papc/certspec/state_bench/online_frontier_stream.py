"""Prefix-local (truly online) STATE-Bench frontier decision.

The offline ``online_residual_selector`` classifies a row from the *whole* saved
cheap trajectory and the runner then slices the pre-frontier prefix. That makes
the strong proposer future-blind, but the *frontier choice itself* is taken with
the full trajectory visible, so it cannot answer "does the method still work when
the frontier must be picked online, before the cheap future exists?".

This module supplies the missing half. A genuinely online controller sees the
cheap conversation grow turn by turn and must decide, using only ``z = prefix``,
whether to freeze ``z_f`` and hand off to the strong residual proposer, or let
the cheap agent keep acting. The strong live continuation (cheap prefix replay,
then live simulator from ``z_f``) already exists in the runner; the only thing
that was offline was the frontier decision.

The decision here is a function of ``conversation[:frontier]`` *only*. We obtain
it by reusing the existing effect-based classifier and re-checking that the same
residual is justified by the pre-frontier prefix in isolation:

1. classify the full row with :func:`online_residual_selector.classify_row`;
2. if it preserves, the online arm preserves (trivially prefix-justified);
3. otherwise rebuild the pre-frontier view (mirroring the runner's
   ``frontier_prefix_payload`` slice) and classify *that* prefix on its own;
4. accept the frontier as online-valid iff the prefix yields the same residual
   class and authority -- i.e. the open obligation is visible from prefix
   evidence, not from the cheap agent's later (possibly wrong) completion.

Two consequences, both intended:

* Effect-based residuals (open state writes detected from the fixed user request
  minus the committed tool-call effects, terminal store-credit writes, ...) are
  justified by the prefix, so they stay online-valid and the frontier handed to
  the runner is byte-identical to the offline one. Offline results are preserved
  by construction on exactly these rows.
* Residuals whose offline detection reads the cheap agent's *post-frontier prose*
  (the lexical protocol-dialogue gates: "can't verify", "couldn't find", an
  out-of-stock disclosure message, ...) are NOT justified by the prefix and are
  reported as ``frontier_requires_post_frontier_evidence``. The online arm
  preserves them for now; they are the precise backlog that de-lexicalising the
  selector (positive, prefix-local, effect-based triggers) must convert.

Like the offline selector this module is scorer-blind: it only ever passes a
whitelisted visible projection into the classifier and never reads scorer fields
or the task id.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from papc.certspec.cfrc_online_runner import (
    PARTIAL_FRONTIER_CONTENT_REDACTED_KEY,
    pre_frontier_view,
)
from papc.certspec.state_bench.online_residual_selector import (
    OnlineCandidateClass,
    OnlineDecision,
    OnlineRowDecision,
    classify_row,
)

SCHEMA = "papc.state_bench_online_frontier_stream.v1"

OnlineRole = Literal["online_residual_candidate", "online_preserve"]

REASON_NO_OFFLINE_FRONTIER = "no_offline_frontier"
REASON_PREFIX_JUSTIFIED = "prefix_justified_frontier"
REASON_REQUIRES_POST_FRONTIER = "frontier_requires_post_frontier_evidence"
REASON_WAITING_FOR_CHEAP_PREFIX = "waiting_for_minimum_cheap_prefix"

# Declarative description of the runtime shape this decision is meant to drive.
# It is public-safe documentation only; it grants no run authorization.
ONLINE_EXECUTION_CONTRACT: dict[str, Any] = {
    "schema": "papc.state_bench_online_frontier_stream_execution_contract.v1",
    "execution_shape": "true_live_cheap_then_cfrc",
    "frontier_decided_on_prefix_only": True,
    "saved_future_suffix_used": False,
    "cheap_rollout_shared_with_baseline": True,
    "strong_takes_over_at_frontier": False,
    "strong_outputs_json_proposal_only": False,
    "strong_outputs_compiler_proposal_only": True,
    "strong_proposal_transport": "native_tool_call",
    "runner_compiler_executes_or_abstains": True,
    "notes": (
        "Frontier is a function of conversation[:frontier] only; the strong "
        "residual model proposes a compiler-gated completion graph, and the runner-owned "
        "compiler decides whether any message or tool call reaches the harness. "
        "Cheap may be run to terminal and reused as both the cheap-only baseline "
        "and the method prefix, because the frontier decision provably never "
        "reads beyond the cut."
    ),
}


class StreamFrontierError(RuntimeError):
    """Raised when a payload cannot be decided online (malformed conversation)."""


@dataclass(frozen=True)
class StreamFrontierDecision:
    """One prefix-local online frontier decision for a single cheap-trajectory row.

    ``frontier_index`` / ``frontier_tool_call_index`` are populated only for an
    online residual candidate and are identical to the coordinates the runner
    already consumes, so this object is a drop-in source of ``z_f`` for the live
    continuation provider.
    """

    row: str
    online_role: OnlineRole
    online_valid: bool
    reason: str
    candidate_class: OnlineCandidateClass
    authority_level: str
    frontier_index: int | None
    frontier_tool_call_index: int | None
    prefix_message_count: int
    offline_decision: OnlineDecision
    offline_candidate_class: OnlineCandidateClass
    class_relevant_families: tuple[str, ...] = ()
    prefix_only: bool = True
    owes_state_write: bool = False

    @property
    def opens_strong_call(self) -> bool:
        return self.online_role == "online_residual_candidate"

    @property
    def diverged_from_offline(self) -> bool:
        """True when the offline arm opened a residual the online arm cannot."""

        return (
            self.offline_decision == "residual_candidate"
            and self.online_role == "online_preserve"
        )

    def to_public_json(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "online_role": self.online_role,
            "online_valid": self.online_valid,
            "reason": self.reason,
            "candidate_class": self.candidate_class,
            "authority_level": self.authority_level,
            "class_relevant_families": list(self.class_relevant_families),
            "frontier_index": self.frontier_index,
            "frontier_tool_call_index": self.frontier_tool_call_index,
            "prefix_message_count": self.prefix_message_count,
            "prefix_only": self.prefix_only,
            "offline_decision": self.offline_decision,
            "offline_candidate_class": self.offline_candidate_class,
            "diverged_from_offline": self.diverged_from_offline,
            "opens_strong_call": self.opens_strong_call,
        }


def decide_online_frontier(
    domain: str,
    payload: Mapping[str, Any],
    *,
    row_index: int,
) -> StreamFrontierDecision:
    """Decide the online frontier for one row using prefix-only evidence."""

    offline = classify_row(domain, payload, row_index=row_index)
    conversation = _conversation(payload)

    if not offline.is_candidate:
        return _preserve(
            offline,
            online_valid=True,
            reason=REASON_NO_OFFLINE_FRONTIER,
            prefix_message_count=len(conversation),
        )

    prefix_view = pre_frontier_view(
        conversation,
        offline.frontier_index,
        offline.frontier_tool_call_index,
    )
    prefix_conversation = prefix_view["conversation"]
    prefix_decision = classify_row(domain, prefix_view, row_index=row_index)

    if _same_residual(offline, prefix_decision):
        return StreamFrontierDecision(
            row=offline.row,
            online_role="online_residual_candidate",
            online_valid=True,
            reason=REASON_PREFIX_JUSTIFIED,
            candidate_class=offline.candidate_class,
            authority_level=offline.authority_level,
            class_relevant_families=_class_relevant_families(offline),
            frontier_index=offline.frontier_index,
            frontier_tool_call_index=offline.frontier_tool_call_index,
            prefix_message_count=len(prefix_conversation),
            offline_decision=offline.decision,
            offline_candidate_class=offline.candidate_class,
            owes_state_write=offline.owes_state_write,
        )

    return _preserve(
        offline,
        online_valid=False,
        reason=REASON_REQUIRES_POST_FRONTIER,
        prefix_message_count=len(prefix_conversation),
    )


def decide_stream_prefix(
    domain: str,
    prefix_payload: Mapping[str, Any],
    *,
    row_index: int,
    min_assistant_turns: int = 1,
) -> StreamFrontierDecision:
    """Decide directly from the currently visible live prefix.

    Unlike :func:`decide_online_frontier`, this function never receives a saved
    cheap future. It is the runtime predicate used after each live cheap step.
    ``min_assistant_turns`` keeps the controller from escalating on the opening
    user request before cheap has had a chance to collect low-authority
    evidence.
    """

    conversation = _conversation(prefix_payload)
    assistant_turns = sum(
        1 for message in conversation if str(message.get("role") or "") == "assistant"
    )
    if assistant_turns < max(0, int(min_assistant_turns)):
        return StreamFrontierDecision(
            row=f"row_{row_index:03d}",
            online_role="online_preserve",
            online_valid=True,
            reason=REASON_WAITING_FOR_CHEAP_PREFIX,
            candidate_class="no_visible_residual",
            authority_level="",
            frontier_index=None,
            frontier_tool_call_index=None,
            prefix_message_count=len(conversation),
            offline_decision="no_frontier",
            offline_candidate_class="no_visible_residual",
        )

    decision = classify_row(domain, prefix_payload, row_index=row_index)
    if not decision.is_candidate:
        return _preserve(
            decision,
            online_valid=True,
            reason=REASON_NO_OFFLINE_FRONTIER,
            prefix_message_count=len(conversation),
        )

    return StreamFrontierDecision(
        row=decision.row,
        online_role="online_residual_candidate",
        online_valid=True,
        reason=REASON_PREFIX_JUSTIFIED,
        candidate_class=decision.candidate_class,
        authority_level=decision.authority_level,
        class_relevant_families=_class_relevant_families(decision),
        frontier_index=decision.frontier_index,
        frontier_tool_call_index=decision.frontier_tool_call_index,
        prefix_message_count=len(conversation),
        offline_decision=decision.decision,
        offline_candidate_class=decision.candidate_class,
        owes_state_write=decision.owes_state_write,
    )


def decide_stream_replay_frontier(
    domain: str,
    payload: Mapping[str, Any],
    *,
    row_index: int,
    min_assistant_turns: int = 1,
) -> StreamFrontierDecision:
    """Replay visible prefixes and return the first true-live frontier.

    This mirrors the controller shape more closely than the single offline cut:
    after every cheap-visible prefix that could precede another assistant turn,
    run the same prefix-local predicate used online. If no prefix opens, fall
    back to the full-row offline classification only to label preserve vs
    de-lexicalization backlog.
    """

    conversation = _conversation(payload)
    for end in range(1, len(conversation) + 1):
        message = conversation[end - 1]
        if _is_task_done_user_message(message):
            break
        decision = decide_stream_prefix(
            domain,
            {"conversation": conversation[:end]},
            row_index=row_index,
            min_assistant_turns=min_assistant_turns,
        )
        if decision.opens_strong_call:
            return decision

    offline = classify_row(domain, payload, row_index=row_index)
    if not offline.is_candidate:
        return _preserve(
            offline,
            online_valid=True,
            reason=REASON_NO_OFFLINE_FRONTIER,
            prefix_message_count=len(conversation),
        )
    return _preserve(
        offline,
        online_valid=False,
        reason=REASON_REQUIRES_POST_FRONTIER,
        prefix_message_count=len(conversation),
    )


def decide_domain_rows(
    domain: str,
    payloads: Sequence[Mapping[str, Any]],
    *,
    replay_stream: bool = False,
) -> list[StreamFrontierDecision]:
    if replay_stream:
        return [
            decide_stream_replay_frontier(domain, payload, row_index=index)
            for index, payload in enumerate(payloads)
        ]
    return [
        decide_online_frontier(domain, payload, row_index=index)
        for index, payload in enumerate(payloads)
    ]


def summarize_stream_decisions(
    rows: Sequence[StreamFrontierDecision],
) -> dict[str, Any]:
    """Public-safe onlineness summary; no task ids, paths, or scorer fields."""

    candidate_classes = Counter(
        row.candidate_class for row in rows if row.opens_strong_call
    )
    authority_counts = Counter(
        row.authority_level
        for row in rows
        if row.opens_strong_call and row.authority_level
    )
    reason_counts = Counter(row.reason for row in rows)
    online_residual_rows = [row for row in rows if row.opens_strong_call]
    diverged_rows = [row for row in rows if row.diverged_from_offline]
    return {
        "schema": SCHEMA,
        "frontier_decision_mode": "prefix_local_stream",
        "frontier_decided_on_prefix_only": True,
        "saved_future_suffix_used": False,
        "row_count": len(rows),
        "online_residual_count": len(online_residual_rows),
        "online_preserve_count": sum(
            1 for row in rows if row.online_role == "online_preserve"
        ),
        "offline_only_residual_count": len(diverged_rows),
        "online_valid_count": sum(1 for row in rows if row.online_valid),
        "predicted_strong_calls": len(online_residual_rows),
        "candidate_class_counts": dict(sorted(candidate_classes.items())),
        "authority_level_counts": dict(sorted(authority_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "needs_delexicalization_rows": sorted(row.row for row in diverged_rows),
    }


def _same_residual(
    offline: OnlineRowDecision,
    prefix_decision: OnlineRowDecision,
) -> bool:
    """Is the offline residual justified by the pre-frontier prefix in isolation?

    The prefix must reproduce the same residual *class*, *authority*, and -- the
    part that separates effect-based from lexical frontiers -- the same specific
    obligation family that drives that class. A user-only prefix can surface a
    generic ``visible_dialogue_residual``/``ell0`` that collides on class and
    authority with a protocol unit detected from post-frontier prose, so the
    class-relevant family tuple must match too.
    """

    return (
        prefix_decision.is_candidate
        and prefix_decision.candidate_class == offline.candidate_class
        and prefix_decision.authority_level == offline.authority_level
        and _class_relevant_families(prefix_decision)
        == _class_relevant_families(offline)
    )


def _class_relevant_families(decision: OnlineRowDecision) -> tuple[str, ...]:
    """The obligation family tuple that drives the decision's candidate class."""

    if decision.candidate_class == "visible_state_write_residual":
        return decision.state_write_families
    if decision.candidate_class == "committed_change_repair_residual":
        return decision.repair_target_families
    if decision.candidate_class == "visible_dialogue_residual":
        return decision.dialogue_families
    return ()


def _preserve(
    offline: OnlineRowDecision,
    *,
    online_valid: bool,
    reason: str,
    prefix_message_count: int,
) -> StreamFrontierDecision:
    return StreamFrontierDecision(
        row=offline.row,
        online_role="online_preserve",
        online_valid=online_valid,
        reason=reason,
        candidate_class="no_visible_residual",
        authority_level="",
        frontier_index=None,
        frontier_tool_call_index=None,
        prefix_message_count=prefix_message_count,
        offline_decision=offline.decision,
        offline_candidate_class=offline.candidate_class,
        owes_state_write=offline.owes_state_write,
    )


def _conversation(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    messages = payload.get("conversation")
    if not isinstance(messages, list):
        messages = payload.get("messages")
    if not isinstance(messages, list):
        raise StreamFrontierError("payload has no conversation/messages list")
    return [message for message in messages if isinstance(message, Mapping)]


def _is_task_done_user_message(message: Mapping[str, Any]) -> bool:
    return str(message.get("role") or "") == "user" and "[TASK_DONE]" in str(
        message.get("content") or ""
    )


__all__ = [
    "ONLINE_EXECUTION_CONTRACT",
    "PARTIAL_FRONTIER_CONTENT_REDACTED_KEY",
    "REASON_NO_OFFLINE_FRONTIER",
    "REASON_PREFIX_JUSTIFIED",
    "REASON_REQUIRES_POST_FRONTIER",
    "REASON_WAITING_FOR_CHEAP_PREFIX",
    "SCHEMA",
    "StreamFrontierDecision",
    "StreamFrontierError",
    "decide_domain_rows",
    "decide_online_frontier",
    "decide_stream_prefix",
    "decide_stream_replay_frontier",
    "pre_frontier_view",
    "summarize_stream_decisions",
]

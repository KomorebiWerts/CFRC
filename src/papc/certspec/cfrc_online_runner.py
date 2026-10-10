"""Benchmark-independent CFRC online runner skeleton.

This module owns method policy that must stay byte-identical across surfaces:
prefix-only enforcement, one-frontier selection, strong proposal invocation,
shared compiler projection, execute/abstain decision, runtime ledger, and the
public/private split. Surface adapters only translate between their local
runtime and the shared CFRC IR.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from papc.certspec.cfrc_compiler import (
    CFRCCompileConfig,
    CFRCCompileResult,
    CFRCFrontier,
    CFRCNodeBinding,
    compile_cfrc_suffix,
    prompt_contract,
)
from papc.certspec.cfrc_ir import ProposalGraph
from papc.certspec.runtime_ledger import RoleCall, RuntimeLedger
from papc.certspec.slot_candidate_generator import SlotConstraints

SCHEMA = "papc.cfrc_online_runner.v1"
PARTIAL_FRONTIER_CONTENT_REDACTED_KEY = "papc_partial_frontier_content_redacted"

RISK_SENTINEL_PROTECTED = "sentinel_protected"
RISK_PREFIX_IRREPARABLE = "prefix_irreparable"
DEFAULT_FORCED_PRESERVE_RISK_FLAGS = (
    RISK_SENTINEL_PROTECTED,
    RISK_PREFIX_IRREPARABLE,
)

_FORBIDDEN_PUBLIC_KEYS = (
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

OnlineCandidateKind = Literal["preserve", "action"]
OnlineDecision = Literal["preserve", "execute", "abstain", "block"]
ProjectionPolicy = str
CompilePolicy = Callable[
    [CFRCFrontier, ProposalGraph, Mapping[str, CFRCNodeBinding], CFRCCompileConfig],
    CFRCCompileResult,
]


class CFRCOnlineSurfaceAdapter(Protocol):
    """Surface-local translation seam for the shared online runner."""

    surface: str

    def cheap_conversation(self) -> Sequence[Mapping[str, Any]]:
        """Return the live/replayed cheap conversation available to the runner."""

    def enumerate_frontier_candidates(
        self, prefix_payload: Mapping[str, Any]
    ) -> Sequence[CFRCOnlineCandidate]:
        """Enumerate preserve/action candidates from the supplied visible prefix."""

    def build_frontier(
        self,
        candidate: CFRCOnlineCandidate,
        prefix_payload: Mapping[str, Any],
    ) -> CFRCFrontier:
        """Translate the selected prefix-local candidate into a verifier frontier."""

    def bind_proposal(
        self,
        candidate: CFRCOnlineCandidate,
        proposal_graph: ProposalGraph,
        prefix_payload: Mapping[str, Any],
    ) -> Mapping[str, CFRCNodeBinding]:
        """Bind the strong proposal graph to executable surface suffix payloads."""

    def execute_suffix(
        self,
        candidate: CFRCOnlineCandidate,
        compile_result: CFRCCompileResult,
    ) -> CFRCOnlineExecution:
        """Execute the compiler-selected suffix in the surface runtime."""


@dataclass(frozen=True)
class CFRCOnlineCandidate:
    """One prefix-local frontier option surfaced by an adapter.

    The candidate deliberately carries no ``CFRCFrontier`` and no proposal graph.
    The runner selects the candidate, cuts the prefix, and only then asks the
    adapter to translate the selected candidate into IR.
    """

    candidate_id: str
    kind: OnlineCandidateKind
    frontier_message_index: int
    frontier_kind: str = ""
    frontier_tool_call_index: int | None = None
    authority_ceiling: str = ""
    preserve_reason: str = ""
    risk_flags: tuple[str, ...] = ()
    effect_units: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in ("preserve", "action"):
            raise ValueError("candidate kind must be preserve or action")
        if not str(self.candidate_id).strip():
            raise ValueError("candidate_id must be non-empty")
        if self.frontier_message_index < 0:
            raise ValueError("frontier_message_index must be >= 0")
        if self.kind == "preserve":
            if not str(self.preserve_reason).strip():
                raise ValueError("preserve candidates must declare preserve_reason")
        else:
            if not str(self.frontier_kind).strip():
                raise ValueError("action candidates must declare frontier_kind")
            if not str(self.authority_ceiling).strip():
                raise ValueError("action candidates must declare authority_ceiling")

    @classmethod
    def preserve(
        cls,
        *,
        candidate_id: str,
        frontier_message_index: int,
        preserve_reason: str,
        risk_flags: Sequence[str] = (),
    ) -> CFRCOnlineCandidate:
        return cls(
            candidate_id=candidate_id,
            kind="preserve",
            frontier_message_index=frontier_message_index,
            preserve_reason=preserve_reason,
            risk_flags=tuple(str(flag) for flag in risk_flags),
        )

    @classmethod
    def action(
        cls,
        *,
        candidate_id: str,
        frontier_kind: str,
        frontier_message_index: int,
        authority_ceiling: str,
        frontier_tool_call_index: int | None = None,
        risk_flags: Sequence[str] = (),
        effect_units: Sequence[str] = (),
    ) -> CFRCOnlineCandidate:
        return cls(
            candidate_id=candidate_id,
            kind="action",
            frontier_kind=frontier_kind,
            frontier_message_index=frontier_message_index,
            frontier_tool_call_index=frontier_tool_call_index,
            authority_ceiling=authority_ceiling,
            risk_flags=tuple(str(flag) for flag in risk_flags),
            effect_units=tuple(str(unit) for unit in effect_units),
        )

    @property
    def is_preserve(self) -> bool:
        return self.kind == "preserve"

    @property
    def is_forced_preserve(self) -> bool:
        return self.is_preserve and bool(
            set(self.risk_flags) & set(DEFAULT_FORCED_PRESERVE_RISK_FLAGS)
        )

    def to_public_json(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "kind": self.kind,
            "frontier_kind": self.frontier_kind,
            "frontier_message_index": self.frontier_message_index,
            "frontier_tool_call_index": self.frontier_tool_call_index,
            "authority_ceiling": self.authority_ceiling,
            "preserve_reason": self.preserve_reason,
            "risk_flags": list(self.risk_flags),
            "effect_unit_count": len(self.effect_units),
            "opens_strong_call": self.kind == "action",
        }


@dataclass(frozen=True)
class CFRCOnlineExecution:
    executed: bool
    public_summary: Mapping[str, Any] = field(default_factory=dict)
    private_summary: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CFRCOnlineProposalContext:
    surface: str
    candidate: CFRCOnlineCandidate
    prefix_payload: Mapping[str, Any]
    frontier: CFRCFrontier
    prompt_contract: Mapping[str, Any]
    compile_config: Mapping[str, Any]


@dataclass(frozen=True)
class CFRCOnlineProposerResult:
    proposal_graph: ProposalGraph
    strong_call: RoleCall | None = None


@dataclass(frozen=True)
class CFRCOnlineResult:
    decision: OnlineDecision
    selected_candidate_id: str
    selected_candidate: CFRCOnlineCandidate
    public_summary: Mapping[str, Any]
    compile_result: CFRCCompileResult | None = None
    execution: CFRCOnlineExecution | None = None


def run_cfrc_online_episode(
    *,
    adapter: CFRCOnlineSurfaceAdapter,
    proposer: Callable[
        [CFRCOnlineProposalContext],
        ProposalGraph | CFRCOnlineProposerResult,
    ],
    compile_config: CFRCCompileConfig,
    strong_call: RoleCall | None = None,
    ledger: RuntimeLedger | None = None,
    authority_order: Sequence[str] | None = None,
    projection_policy: ProjectionPolicy = "full",
    compile_policy: CompilePolicy | None = None,
) -> CFRCOnlineResult:
    """Run one online CFRC episode through the shared skeleton."""

    projection_policy = str(projection_policy).strip()
    if not projection_policy:
        raise ValueError("projection policy label must be non-empty")
    if projection_policy == "full" and compile_policy is not None:
        raise ValueError("full projection cannot use a surface compile policy")
    if projection_policy != "full" and compile_policy is None:
        raise ValueError("non-full projection requires a surface compile policy")
    runtime_ledger = ledger or RuntimeLedger()
    conversation = [
        copy.deepcopy(dict(message)) for message in adapter.cheap_conversation()
    ]
    full_payload = {"conversation": conversation}
    raw_candidates = _normalized_candidates(
        adapter.enumerate_frontier_candidates(full_payload)
    )
    prefix_payload_by_id, prefix_rejected = _prefix_justified_action_payloads(
        adapter=adapter,
        conversation=conversation,
        candidates=raw_candidates,
    )
    eligible_candidates = [
        candidate
        for candidate in raw_candidates
        if candidate.is_preserve or candidate.candidate_id in prefix_payload_by_id
    ]
    selected = _select_candidate(
        eligible_candidates,
        authority_order=authority_order or _authority_order(compile_config),
    )

    if selected.is_preserve:
        summary = _base_public_summary(
            surface=adapter.surface,
            selected=selected,
            raw_candidates=raw_candidates,
            prefix_rejected=prefix_rejected,
            strong_calls=0,
            runtime_ledger=runtime_ledger,
            compile_result=None,
            execution=None,
            decision="preserve",
            projection_policy=projection_policy,
        )
        assert_public_summary_safe(summary)
        return CFRCOnlineResult(
            decision="preserve",
            selected_candidate_id=selected.candidate_id,
            selected_candidate=selected,
            public_summary=summary,
        )

    prefix_payload = prefix_payload_by_id[selected.candidate_id]
    frontier = adapter.build_frontier(selected, prefix_payload)
    context = CFRCOnlineProposalContext(
        surface=adapter.surface,
        candidate=selected,
        prefix_payload=prefix_payload,
        frontier=frontier,
        prompt_contract=_proposer_prompt_contract(
            adapter=adapter,
            candidate=selected,
            frontier=frontier,
            prefix_payload=prefix_payload,
            compile_config=compile_config,
        ),
        compile_config=compile_config.to_json(),
    )
    proposer_result = _normalize_proposer_result(proposer(context))
    proposal_graph = proposer_result.proposal_graph
    observed_strong_call = (
        proposer_result.strong_call
        or strong_call
        or RoleCall(role="strong", token_usage_available=False)
    )
    runtime_ledger.add(observed_strong_call)
    bindings = adapter.bind_proposal(selected, proposal_graph, prefix_payload)
    slot_constraints = _slot_constraints(
        adapter, selected, proposal_graph, prefix_payload
    )
    compile_result = (
        compile_cfrc_suffix(
            frontier=frontier,
            proposal_graph=proposal_graph,
            bindings=bindings,
            config=compile_config,
            slot_constraints=slot_constraints,
        )
        if compile_policy is None
        else compile_policy(frontier, proposal_graph, bindings, compile_config)
    )
    execution = (
        adapter.execute_suffix(selected, compile_result)
        if compile_result.decision == "execute"
        else None
    )
    decision: OnlineDecision = compile_result.decision
    summary = _base_public_summary(
        surface=adapter.surface,
        selected=selected,
        raw_candidates=raw_candidates,
        prefix_rejected=prefix_rejected,
        strong_calls=max(0, int(observed_strong_call.call_count)),
        runtime_ledger=runtime_ledger,
        compile_result=compile_result,
        execution=execution,
        decision=decision,
        projection_policy=projection_policy,
    )
    assert_public_summary_safe(summary)
    return CFRCOnlineResult(
        decision=decision,
        selected_candidate_id=selected.candidate_id,
        selected_candidate=selected,
        compile_result=compile_result,
        execution=execution,
        public_summary=summary,
    )


def _compile_proposal(
    *,
    adapter: CFRCOnlineSurfaceAdapter,
    selected: CFRCOnlineCandidate,
    prefix_payload: Mapping[str, Any],
    frontier: CFRCFrontier,
    proposal_graph: ProposalGraph,
    bindings: Mapping[str, CFRCNodeBinding],
    compile_config: CFRCCompileConfig,
    compile_policy: CompilePolicy | None,
) -> CFRCCompileResult:
    slot_constraints = _slot_constraints(
        adapter, selected, proposal_graph, prefix_payload
    )
    return (
        compile_cfrc_suffix(
            frontier=frontier,
            proposal_graph=proposal_graph,
            bindings=bindings,
            config=compile_config,
            slot_constraints=slot_constraints,
        )
        if compile_policy is None
        else compile_policy(frontier, proposal_graph, bindings, compile_config)
    )


def pre_frontier_view(
    conversation: Sequence[Mapping[str, Any]],
    frontier_index: int,
    frontier_tool_call_index: int | None = None,
) -> dict[str, Any]:
    """Return the visible prefix before a frontier action.

    If the frontier is inside a multi-tool assistant turn, read calls before the
    write are preserved while the assistant summary text is redacted as future
    cheap output.
    """

    messages = list(conversation)
    cut = max(0, min(int(frontier_index), len(messages)))
    prefix = [copy.deepcopy(message) for message in messages[:cut]]
    tool_cut = _normalized_tool_call_cut(messages, cut, frontier_tool_call_index)
    if tool_cut is not None and tool_cut > 0 and cut < len(messages):
        message = messages[cut]
        if isinstance(message, Mapping):
            partial = copy.deepcopy(dict(message))
            partial["tool_calls"] = [
                copy.deepcopy(call)
                for call in list(message.get("tool_calls") or [])[:tool_cut]
                if isinstance(call, Mapping)
            ]
            if str(partial.get("content") or "").strip():
                partial["content"] = ""
                partial[PARTIAL_FRONTIER_CONTENT_REDACTED_KEY] = True
            prefix.append(partial)
    return {"conversation": prefix}


def assert_public_summary_safe(value: Any) -> None:
    """Fail if a public summary exposes forbidden benchmark/private fields."""

    _scan_public_value(value, path="summary")


def _prefix_justified_action_payloads(
    *,
    adapter: CFRCOnlineSurfaceAdapter,
    conversation: Sequence[Mapping[str, Any]],
    candidates: Sequence[CFRCOnlineCandidate],
) -> tuple[dict[str, Mapping[str, Any]], int]:
    prefix_payload_by_id: dict[str, Mapping[str, Any]] = {}
    rejected = 0
    for candidate in candidates:
        if candidate.is_preserve:
            continue
        prefix_payload = pre_frontier_view(
            conversation,
            candidate.frontier_message_index,
            candidate.frontier_tool_call_index,
        )
        prefix_candidates = _normalized_candidates(
            adapter.enumerate_frontier_candidates(prefix_payload)
        )
        if _action_reproduced(candidate, prefix_candidates):
            prefix_payload_by_id[candidate.candidate_id] = prefix_payload
        else:
            rejected += 1
    return prefix_payload_by_id, rejected


def _action_reproduced(
    candidate: CFRCOnlineCandidate,
    prefix_candidates: Sequence[CFRCOnlineCandidate],
) -> bool:
    return any(
        other.kind == "action"
        and other.candidate_id == candidate.candidate_id
        and other.frontier_kind == candidate.frontier_kind
        and other.authority_ceiling == candidate.authority_ceiling
        and other.effect_units == candidate.effect_units
        for other in prefix_candidates
    )


def _select_candidate(
    candidates: Sequence[CFRCOnlineCandidate],
    *,
    authority_order: Sequence[str],
) -> CFRCOnlineCandidate:
    if not candidates:
        raise ValueError("frontier portfolio must offer at least one candidate")
    preserves = [candidate for candidate in candidates if candidate.is_preserve]
    if not preserves:
        raise ValueError("frontier portfolio must include a preserve floor")
    forced = [candidate for candidate in preserves if candidate.is_forced_preserve]
    if forced:
        return min(forced, key=lambda candidate: candidate.candidate_id)
    actions = [candidate for candidate in candidates if not candidate.is_preserve]
    if not actions:
        return min(preserves, key=lambda candidate: candidate.candidate_id)
    authority_rank = {label: index for index, label in enumerate(authority_order)}
    return min(
        actions,
        key=lambda candidate: (
            authority_rank.get(candidate.authority_ceiling, len(authority_rank)),
            candidate.frontier_kind,
            candidate.candidate_id,
        ),
    )


def _base_public_summary(
    *,
    surface: str,
    selected: CFRCOnlineCandidate,
    raw_candidates: Sequence[CFRCOnlineCandidate],
    prefix_rejected: int,
    strong_calls: int,
    runtime_ledger: RuntimeLedger,
    compile_result: CFRCCompileResult | None,
    execution: CFRCOnlineExecution | None,
    decision: OnlineDecision,
    projection_policy: ProjectionPolicy,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "surface": surface,
        "decision": decision,
        "projection_policy": projection_policy,
        "selected_candidate_id": selected.candidate_id,
        "selected_candidate": selected.to_public_json(),
        "frontier_decided_on_prefix_only": True,
        "saved_future_suffix_used": False,
        "candidate_count": len(raw_candidates),
        "prefix_only_rejected_action_count": prefix_rejected,
        "strong_calls": strong_calls,
        "compile_result": compile_result.to_json()
        if compile_result is not None
        else None,
        "execution": dict(execution.public_summary) if execution is not None else {},
        "runtime_ledger": runtime_ledger.public_summary(),
        "public_safe": True,
    }


def _normalized_candidates(
    candidates: Sequence[CFRCOnlineCandidate],
) -> tuple[CFRCOnlineCandidate, ...]:
    normalized = tuple(candidates)
    ids = [candidate.candidate_id for candidate in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError("frontier candidate ids must be unique")
    return normalized


def _normalize_proposer_result(
    result: ProposalGraph | CFRCOnlineProposerResult,
) -> CFRCOnlineProposerResult:
    if isinstance(result, CFRCOnlineProposerResult):
        return result
    if isinstance(result, ProposalGraph):
        return CFRCOnlineProposerResult(proposal_graph=result)
    raise TypeError("proposer must return ProposalGraph or CFRCOnlineProposerResult")


def _slot_constraints(
    adapter: CFRCOnlineSurfaceAdapter,
    candidate: CFRCOnlineCandidate,
    proposal_graph: ProposalGraph,
    prefix_payload: Mapping[str, Any],
) -> SlotConstraints | None:
    method = getattr(adapter, "slot_constraints", None)
    if method is None:
        return None
    return method(candidate, proposal_graph, prefix_payload)


def _proposer_prompt_contract(
    *,
    adapter: CFRCOnlineSurfaceAdapter,
    candidate: CFRCOnlineCandidate,
    frontier: CFRCFrontier,
    prefix_payload: Mapping[str, Any],
    compile_config: CFRCCompileConfig,
) -> dict[str, Any]:
    contract = prompt_contract(frontier, compile_config).to_json()
    method = getattr(adapter, "proposer_prompt_contract", None)
    if method is None:
        return contract
    extra = method(candidate, frontier, prefix_payload)
    if isinstance(extra, Mapping):
        contract.update(copy.deepcopy(dict(extra)))
    return contract


def _authority_order(config: CFRCCompileConfig) -> tuple[str, ...]:
    return tuple(level.label for level in config.authority_lattice.levels)


def _normalized_tool_call_cut(
    messages: Sequence[Any],
    frontier_index: int,
    frontier_tool_call_index: int | None,
) -> int | None:
    if frontier_tool_call_index is None:
        return None
    if frontier_index < 0 or frontier_index >= len(messages):
        return None
    message = messages[frontier_index]
    if not isinstance(message, Mapping):
        return None
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return None
    try:
        tool_cut = int(frontier_tool_call_index)
    except (TypeError, ValueError):
        return None
    return max(0, min(tool_cut, len(calls)))


def _scan_public_value(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            key_token = _tokenize(key_text)
            for forbidden in _FORBIDDEN_PUBLIC_KEYS:
                if forbidden in key_token:
                    raise ValueError(
                        f"public summary contains forbidden key {forbidden} at {path}"
                    )
            _scan_public_value(child, path=f"{path}.{key_text}")
        return
    if isinstance(value, (list, tuple, set)):
        for index, child in enumerate(value):
            _scan_public_value(child, path=f"{path}[{index}]")


def _tokenize(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


__all__ = [
    "DEFAULT_FORCED_PRESERVE_RISK_FLAGS",
    "PARTIAL_FRONTIER_CONTENT_REDACTED_KEY",
    "RISK_PREFIX_IRREPARABLE",
    "RISK_SENTINEL_PROTECTED",
    "SCHEMA",
    "CFRCOnlineCandidate",
    "CFRCOnlineExecution",
    "CFRCOnlineProposalContext",
    "CFRCOnlineProposerResult",
    "CFRCOnlineResult",
    "CFRCOnlineSurfaceAdapter",
    "CompilePolicy",
    "ProjectionPolicy",
    "assert_public_summary_safe",
    "pre_frontier_view",
    "run_cfrc_online_episode",
]

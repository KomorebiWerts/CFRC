"""True-online ToolSandbox CFRC controller.

The controller owns ToolSandbox episode stepping only.  Method policy remains
in the shared ``run_cfrc_online_episode`` skeleton: candidate selection,
prefix-only handoff, the strong proposal call, compiler projection, execution
decision, and the public ledger.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from papc.certspec.cfrc_compiler import (
    CFRCCompileConfig,
    CFRCNodeBinding,
)
from papc.certspec.cfrc_ir import ProposalGraph
from papc.certspec.cfrc_online_runner import (
    CFRCOnlineCandidate,
    CFRCOnlineExecution,
    CFRCOnlineProposerResult,
    CFRCOnlineResult,
    assert_public_summary_safe,
    run_cfrc_online_episode,
)
from papc.certspec.runtime_ledger import RoleCall, RuntimeLedger
from papc.certspec.toolsandbox.cfrc_online_adapter import (
    ToolSandboxTypedSuffixOnlineAdapter,
)
from papc.certspec.toolsandbox.grounded_residual_authoring import (
    build_grounded_bundle,
    run_grounded_authoring,
)
from papc.certspec.toolsandbox.online_strong_proposer import EMPTY_GRAPH_ID
from papc.certspec.toolsandbox.public_hygiene import public_leakage_audit

SCHEMA = "papc.toolsandbox.online_cfrc_controller.v1"
PRIVATE_SCHEMA = "papc.toolsandbox.online_cfrc_controller.private_receipt.v1"


class ToolSandboxOnlineRuntime(Protocol):
    """Runtime seam for a single live ToolSandbox episode."""

    surface: str

    def start_episode(self, scenario_name: str) -> None:
        """Start from the official scenario initial context."""

    def is_active(self) -> bool:
        """Return whether the ToolSandbox conversation can advance."""

    def step(self) -> Mapping[str, Any]:
        """Advance one official role transition."""

    def visible_conversation(self) -> Sequence[Mapping[str, Any]]:
        """Return only the currently visible prefix."""

    def available_tool_names(self) -> Sequence[str]:
        """Return the public tool names exposed in the current episode."""

    def snapshot_context(self) -> Any:
        """Return an independent checkpoint of the reached native state."""

    def restore_context(self, checkpoint: Any) -> None:
        """Restore one native checkpoint as the active state."""

    def context_sha256(self, checkpoint: Any | None = None) -> str:
        """Return a deterministic digest for one native state."""

    def context_replay_sha256(self, reached_checkpoint: Any) -> str:
        """Digest post-state modulo native IDs allocated after the checkpoint."""

    def execute_grounded_call(
        self,
        *,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Execute one native call and return its receipt for grounded authoring."""

    def execute_suffix(self, suffix_payloads: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Execute compiler-selected residual suffix through ToolSandbox runtime."""

    def evaluate(self) -> Mapping[str, Any]:
        """Run the official evaluator on the current live episode state."""


class ToolSandboxFrontierMonitor(Protocol):
    """Deterministic monitor that enumerates one typed row from live prefix state."""

    def typed_row_from_context(self, context: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """Return an actionable ToolSandbox typed row, or ``None`` to keep cheap running."""


# One authoring step: the controller hands the successor the visible prefix, the
# authorized residual, and every receipt observed so far; the successor returns
# the next native call, the terminal message, or a discharge declaration.
StrongProposer = Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class ToolSandboxOnlineCFRCConfig:
    """Per-episode controls for the true-online ToolSandbox method."""

    max_turn_steps: int
    # Remaining provider-call budget, shared across episodes by the runner.
    max_strong_calls: int = 1
    # How many frontiers one episode may open. Receipt-grounded authoring spends
    # several provider calls per frontier, so the two limits can no longer be the
    # same number.
    max_frontier_attempts: int = 1
    max_authoring_rounds: int = 14
    max_reproposal_rounds: int = 2
    terminal_with_last_call: bool = False
    surface: str = "toolsandbox"
    paper_profile: bool = True

    def __post_init__(self) -> None:
        if int(self.max_turn_steps) <= 0:
            raise ValueError("max_turn_steps must be positive")
        if int(self.max_strong_calls) < 0:
            raise ValueError("max_strong_calls must be non-negative")
        if int(self.max_frontier_attempts) <= 0:
            raise ValueError("max_frontier_attempts must be positive")
        if int(self.max_authoring_rounds) <= 0:
            raise ValueError("max_authoring_rounds must be positive")
        if int(self.max_reproposal_rounds) <= 0:
            raise ValueError("max_reproposal_rounds must be positive")
        if self.surface != "toolsandbox":
            raise ValueError("ToolSandboxOnlineCFRCConfig surface must be toolsandbox")
        if type(self.paper_profile) is not bool:
            raise TypeError("paper_profile must be bool")
        if type(self.terminal_with_last_call) is not bool:
            raise TypeError("terminal_with_last_call must be bool")

@dataclass
class ToolSandboxOnlineCFRCController:
    """Run cheap live until an online frontier, then hand prefix to CFRC skeleton."""

    runtime: ToolSandboxOnlineRuntime
    frontier_monitor: ToolSandboxFrontierMonitor
    strong_proposer: StrongProposer
    compile_config: CFRCCompileConfig

    @property
    def _declarations_accepted(self) -> bool:
        """Report whether this contract can discharge an obligation without a receipt.

        Requiring an action receipt witness is exactly the statement that it
        cannot. Deriving the answer here keeps the successor's stated options and
        the checker's verdict from drifting apart.
        """

        return not bool(self.compile_config.require_action_receipt_witness)

    def run_episode(
        self,
        scenario_name: str,
        config: ToolSandboxOnlineCFRCConfig,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run one true-online ToolSandbox episode.

        ``scenario_name`` stays private.  The public summary intentionally does
        not include the scenario id, raw prefix content, task id, gold state, or
        scorer payload.
        """

        episode_started = time.monotonic()
        self.runtime.start_episode(scenario_name)
        cheap_steps = 0
        cfrc_attempts: list[_CFRCAttempt] = []
        resumed_after_block = False
        execute_payload: Mapping[str, Any] | None = None
        final_evaluation: Mapping[str, Any] | None = None
        frontier_seen = False
        frontier_suppressed_by_strong_cap = False
        strong_call_count = 0
        frontier_attempts = 0
        method_refused = False

        initial_prefix = self._visible_prefix()
        if initial_prefix:
            typed_row = self.frontier_monitor.typed_row_from_context(
                self._monitor_context(
                    conversation=initial_prefix,
                    cheap_step_count=cheap_steps,
                )
            )
            if typed_row is not None:
                frontier_seen = True
                if _frontier_suppressed(
                    frontier_attempts=frontier_attempts,
                    strong_call_count=strong_call_count,
                    config=config,
                ):
                    frontier_suppressed_by_strong_cap = True
                else:
                    frontier_attempts += 1
                    result, round_attempts = self._run_frontier_rounds(
                        prefix=initial_prefix,
                        typed_row=typed_row,
                        config=config,
                        strong_calls_used=strong_call_count,
                    )
                    strong_call_count += sum(
                        attempt.strong_calls for attempt in round_attempts
                    )
                    cfrc_attempts.extend(round_attempts)
                    if _execution_committed(result):
                        execute_payload = result.execution.public_summary
                        final_evaluation = _evaluation_payload(execute_payload)
                    else:
                        if config.paper_profile:
                            method_refused = True
                        else:
                            resumed_after_block = result.decision in {
                                "abstain",
                                "block",
                                "preserve",
                            }

        while (
            execute_payload is None
            and not method_refused
            and self.runtime.is_active()
            and cheap_steps < int(config.max_turn_steps)
        ):
            step_result = self.runtime.step()
            if step_result.get("advanced") is False:
                break
            cheap_steps += 1

            prefix = self._visible_prefix()
            typed_row = self.frontier_monitor.typed_row_from_context(
                self._monitor_context(
                    conversation=prefix,
                    cheap_step_count=cheap_steps,
                )
            )
            if typed_row is None:
                continue

            frontier_seen = True
            if _frontier_suppressed(
                frontier_attempts=frontier_attempts,
                strong_call_count=strong_call_count,
                config=config,
            ):
                frontier_suppressed_by_strong_cap = True
                continue

            frontier_attempts += 1
            result, round_attempts = self._run_frontier_rounds(
                prefix=prefix,
                typed_row=typed_row,
                config=config,
                strong_calls_used=strong_call_count,
            )
            strong_call_count += sum(
                attempt.strong_calls for attempt in round_attempts
            )
            cfrc_attempts.extend(round_attempts)

            if _execution_committed(result):
                execute_payload = result.execution.public_summary
                final_evaluation = _evaluation_payload(execute_payload)
                break

            if config.paper_profile:
                method_refused = True
                break
            resumed_after_block = result.decision in {"abstain", "block", "preserve"}

        if execute_payload is not None:
            decision = "execute"
        elif method_refused:
            final_evaluation = self.runtime.evaluate()
            decision = "cfrc_refused_no_handback"
        else:
            final_evaluation = self.runtime.evaluate()
            if resumed_after_block:
                decision = "cheap_continued_after_strong_block"
            elif frontier_suppressed_by_strong_cap:
                decision = "cheap_completed_after_strong_cap_exhausted_frontier"
            else:
                decision = "cheap_completed_without_frontier"

        method_wall_clock_seconds = round(time.monotonic() - episode_started, 6)
        runtime_ledger = _merged_runtime_ledger(self.runtime, cfrc_attempts)
        public_summary = self._public_summary(
            decision=decision,
            cheap_steps=cheap_steps,
            frontier_seen=frontier_seen,
            frontier_suppressed_by_strong_cap=frontier_suppressed_by_strong_cap,
            resumed_after_block=resumed_after_block,
            method_refused=method_refused,
            paper_profile=config.paper_profile,
            strong_call_count=strong_call_count,
            cfrc_attempts=cfrc_attempts,
            final_evaluation=final_evaluation,
            runtime_ledger=runtime_ledger,
            method_wall_clock_seconds=method_wall_clock_seconds,
            terminal_with_last_call=config.terminal_with_last_call,
        )
        private_receipt = self._private_receipt(
            scenario_name=scenario_name,
            cheap_steps=cheap_steps,
            strong_call_count=strong_call_count,
            frontier_suppressed_by_strong_cap=frontier_suppressed_by_strong_cap,
            method_refused=method_refused,
            paper_profile=config.paper_profile,
            cfrc_attempts=cfrc_attempts,
            public_summary=public_summary,
            runtime_ledger=runtime_ledger,
            method_wall_clock_seconds=method_wall_clock_seconds,
            terminal_with_last_call=config.terminal_with_last_call,
        )
        return private_receipt, public_summary

    def _run_frontier_rounds(
        self,
        *,
        prefix: Sequence[Mapping[str, Any]],
        typed_row: Mapping[str, Any],
        config: ToolSandboxOnlineCFRCConfig,
        strong_calls_used: int,
    ) -> tuple[CFRCOnlineResult, list["_CFRCAttempt"]]:
        """Author, check, and re-propose within a bounded round budget.

        A refusal is only final once the budget is spent.  Until then the
        compiler's reasons and the replica's failures go back to the successor as
        typed feedback, which is what makes the check three-valued rather than a
        single pass/fail gate.  Exhausting the budget still returns control to
        nobody: the method refuses instead of handing back to the cheap agent.

        Each authoring step is its own provider call, so the shared budget has to
        bound the steps as well as the rounds.  Checking it only before the first
        round would let one frontier overspend the whole run.
        """

        feedback: list[Mapping[str, Any]] = []
        attempts: list[_CFRCAttempt] = []
        result: CFRCOnlineResult | None = None
        spent = 0
        for round_index in range(1, int(config.max_reproposal_rounds) + 1):
            remaining = int(config.max_strong_calls) - int(strong_calls_used) - spent
            if remaining <= 0:
                break
            result, diagnostic = self._run_cfrc_attempt(
                prefix=prefix,
                typed_row=typed_row,
                feedback=feedback,
                max_authoring_rounds=min(int(config.max_authoring_rounds), remaining),
                terminal_with_last_call=config.terminal_with_last_call,
            )
            attempt = _CFRCAttempt.from_result(result, proposer_diagnostic=diagnostic)
            attempts.append(attempt)
            spent += attempt.strong_calls
            if _execution_committed(result):
                break
            feedback.append(
                _typed_verifier_feedback(
                    round_index=round_index,
                    result=result,
                    diagnostic=diagnostic,
                )
            )
        if result is None:
            raise RuntimeError("frontier round budget produced no CFRC result")
        return result, attempts

    def _run_cfrc_attempt(
        self,
        *,
        prefix: Sequence[Mapping[str, Any]],
        typed_row: Mapping[str, Any],
        feedback: Sequence[Mapping[str, Any]],
        max_authoring_rounds: int,
        terminal_with_last_call: bool,
    ) -> tuple[CFRCOnlineResult, Mapping[str, Any]]:
        grounded_bindings: dict[str, CFRCNodeBinding] = {}
        proposer_diagnostic: dict[str, Any] = {}
        adapter = _RuntimeExecutingToolSandboxAdapter(
            cheap_messages=prefix,
            typed_row=typed_row,
            runtime=self.runtime,
            grounded_bindings=grounded_bindings,
        )
        result = run_cfrc_online_episode(
            adapter=adapter,
            proposer=self._grounded_proposer(
                typed_row=typed_row,
                feedback=feedback,
                max_authoring_rounds=max_authoring_rounds,
                grounded_bindings=grounded_bindings,
                diagnostic=proposer_diagnostic,
                terminal_with_last_call=terminal_with_last_call,
            ),
            compile_config=self.compile_config,
        )
        return result, proposer_diagnostic

    def _grounded_proposer(
        self,
        *,
        typed_row: Mapping[str, Any],
        feedback: Sequence[Mapping[str, Any]],
        max_authoring_rounds: int,
        grounded_bindings: dict[str, CFRCNodeBinding],
        diagnostic: dict[str, Any],
        terminal_with_last_call: bool,
    ) -> Callable[[Any], CFRCOnlineProposerResult]:
        def propose(context: Any) -> CFRCOnlineProposerResult:
            prefix_conversation = [
                message
                for message in context.prefix_payload.get("conversation") or ()
                if isinstance(message, Mapping)
            ]
            # Authoring runs against a replica: the successor must observe real
            # receipts, and the reached state must be untouched when it ends.
            checkpoint = self.runtime.snapshot_context()
            try:
                outcome = run_grounded_authoring(
                    runtime=self.runtime,
                    typed_row=typed_row,
                    prefix_conversation=prefix_conversation,
                    next_step=self.strong_proposer,
                    max_authoring_rounds=max_authoring_rounds,
                    feedback=feedback,
                    declarations_accepted=self._declarations_accepted,
                    terminal_with_last_call=terminal_with_last_call,
                )
            finally:
                self.runtime.restore_context(checkpoint)
            bundle, blockers = build_grounded_bundle(
                outcome=outcome,
                typed_row=typed_row,
                prefix_conversation=prefix_conversation,
            )
            grounded_bindings.clear()
            if bundle is not None:
                grounded_bindings.update(dict(bundle.bindings))
            # Draining publishes this pass's step records. Reading the proposer's
            # diagnostic before the drain would record the previous pass, and the
            # last pass of a frontier would never be recorded at all.
            strong_call = _drained_strong_call(self.strong_proposer)
            diagnostic.clear()
            diagnostic.update(
                {
                    "authoring": outcome.public_summary(),
                    "grounded_blockers": list(blockers),
                    "grounded_proposal": dict(bundle.public_summary)
                    if bundle is not None
                    else {},
                    "step_proposer": dict(
                        getattr(self.strong_proposer, "latest_private_diagnostic", {})
                        or {}
                    ),
                }
            )
            return CFRCOnlineProposerResult(
                proposal_graph=bundle.proposal_graph
                if bundle is not None
                else ProposalGraph(nodes=(), graph_id=EMPTY_GRAPH_ID),
                strong_call=strong_call,
            )

        return propose

    def _visible_prefix(self) -> list[Mapping[str, Any]]:
        return [copy.deepcopy(dict(message)) for message in self.runtime.visible_conversation()]

    def _monitor_context(
        self,
        *,
        conversation: Sequence[Mapping[str, Any]],
        cheap_step_count: int,
    ) -> dict[str, Any]:
        context: dict[str, Any] = {
            "surface": "toolsandbox",
            "conversation": conversation,
            "cheap_step_count": int(cheap_step_count),
        }
        available_tool_names = getattr(self.runtime, "available_tool_names", None)
        if callable(available_tool_names):
            context["available_tool_names"] = [
                str(name) for name in available_tool_names()
            ]
        return context

    def _public_summary(
        self,
        *,
        decision: str,
        cheap_steps: int,
        frontier_seen: bool,
        frontier_suppressed_by_strong_cap: bool,
        resumed_after_block: bool,
        method_refused: bool,
        paper_profile: bool,
        strong_call_count: int,
        cfrc_attempts: Sequence["_CFRCAttempt"],
        final_evaluation: Mapping[str, Any] | None,
        runtime_ledger: Mapping[str, Any],
        method_wall_clock_seconds: float,
        terminal_with_last_call: bool,
    ) -> dict[str, Any]:
        evaluation = _public_evaluation(final_evaluation)
        checked_execution_attempted = any(
            _attempt_execution(attempt).get("replica_execution_performed") is True
            for attempt in cfrc_attempts
        )
        checked_execution_admitted = any(
            _attempt_execution(attempt).get("checked_execution_admitted") is True
            for attempt in cfrc_attempts
        )
        # The contract asks for an answer to the user, so whether one was actually
        # submitted is part of the outcome. Reporting only the tool calls hides a
        # residual that changed state and told the user nothing.
        terminal_answer_submitted = any(
            _nonnegative_int(
                _attempt_execution(attempt).get("assistant_response_appended_count")
            )
            > 0
            for attempt in cfrc_attempts
        )
        summary: dict[str, Any] = {
            "schema": SCHEMA,
            "surface": "toolsandbox",
            "mode": "toolsandbox_true_online_cfrc",
            "decision": decision,
            "cheap_step_count": int(cheap_steps),
            "online_frontier_generated": bool(frontier_seen),
            "frontier_suppressed_by_strong_cap": bool(
                frontier_suppressed_by_strong_cap
            ),
            "strong_call_count": int(strong_call_count),
            "method_wall_clock_seconds": float(method_wall_clock_seconds),
            "cfrc_attempt_count": len(cfrc_attempts),
            "cfrc_attempt_decisions": [attempt.decision for attempt in cfrc_attempts],
            "grounded_authoring": _grounded_authoring_summary(cfrc_attempts),
            "resumed_cheap_after_block": bool(resumed_after_block),
            "method_refused": bool(method_refused),
            "paper_profile": bool(paper_profile),
            "unchecked_handback_used": bool(resumed_after_block),
            "checked_execution_attempted": checked_execution_attempted,
            "checked_execution_admitted": checked_execution_admitted,
            "checked_execution_replay_match": checked_execution_admitted,
            "terminal_answer_submitted": terminal_answer_submitted,
            "cheap_preserved": decision == "cheap_completed_without_frontier",
            "used_completed_lower_replay": False,
            "saved_future_suffix_used": False,
            "full_agent_fallback_used": False,
            "official_evaluator_used": bool(
                final_evaluation and final_evaluation.get("official_evaluator_used") is True
            ),
            "evaluation": evaluation,
            "method_boundary": {
                "shared_skeleton_owned_decision": True,
                "surface_adapter_enumerated_frontier_only": True,
                "prefix_only_strong_handoff": True,
            },
            "runtime_ledger": copy.deepcopy(dict(runtime_ledger)),
            "cfrc": [attempt.public_summary for attempt in cfrc_attempts],
        }
        if terminal_with_last_call:
            summary["method_variant"] = "terminal_with_last_call"
        assert_public_summary_safe(summary)
        leakage = public_leakage_audit(summary)
        summary["public_leakage_audit"] = leakage
        if leakage["pass"] is not True:
            raise ValueError("ToolSandbox online public summary failed leakage audit")
        return summary

    def _private_receipt(
        self,
        *,
        scenario_name: str,
        cheap_steps: int,
        strong_call_count: int,
        frontier_suppressed_by_strong_cap: bool,
        method_refused: bool,
        paper_profile: bool,
        cfrc_attempts: Sequence["_CFRCAttempt"],
        public_summary: Mapping[str, Any],
        runtime_ledger: Mapping[str, Any],
        method_wall_clock_seconds: float,
        terminal_with_last_call: bool,
    ) -> dict[str, Any]:
        receipt = {
            "schema": PRIVATE_SCHEMA,
            "surface": "toolsandbox",
            "mode": "toolsandbox_true_online_cfrc_private",
            "private": {
                "scenario_name": scenario_name,
            },
            "cheap_step_count": int(cheap_steps),
            "strong_call_count": int(strong_call_count),
            "method_wall_clock_seconds": float(method_wall_clock_seconds),
            "frontier_suppressed_by_strong_cap": bool(
                frontier_suppressed_by_strong_cap
            ),
            "method_refused": bool(method_refused),
            "paper_profile": bool(paper_profile),
            "runtime_ledger": copy.deepcopy(dict(runtime_ledger)),
            "cfrc_attempts": [attempt.to_private_json() for attempt in cfrc_attempts],
            "public_decision": public_summary["decision"],
            "public_summary_digest_ready": True,
        }
        if terminal_with_last_call:
            receipt["method_variant"] = "terminal_with_last_call"
        return receipt


@dataclass(frozen=True)
class _CFRCAttempt:
    decision: str
    selected_candidate_id: str
    strong_calls: int
    public_summary: Mapping[str, Any] = field(default_factory=dict)
    proposer_diagnostic: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_result(
        cls,
        result: CFRCOnlineResult,
        *,
        proposer_diagnostic: Mapping[str, Any] | None = None,
    ) -> "_CFRCAttempt":
        return cls(
            decision=result.decision,
            selected_candidate_id=result.selected_candidate_id,
            strong_calls=_strong_calls(result),
            public_summary=copy.deepcopy(dict(result.public_summary)),
            proposer_diagnostic=copy.deepcopy(dict(proposer_diagnostic or {})),
        )

    def to_private_json(self) -> dict[str, Any]:
        payload = {
            "decision": self.decision,
            "selected_candidate_id": self.selected_candidate_id,
            "strong_calls": self.strong_calls,
            "public_summary": copy.deepcopy(dict(self.public_summary)),
        }
        if self.proposer_diagnostic:
            payload["proposer_diagnostic"] = copy.deepcopy(
                dict(self.proposer_diagnostic)
            )
        return payload


@dataclass
class _RuntimeExecutingToolSandboxAdapter:
    cheap_messages: Sequence[Mapping[str, Any]]
    typed_row: Mapping[str, Any]
    runtime: ToolSandboxOnlineRuntime
    grounded_bindings: Mapping[str, CFRCNodeBinding] | None = None
    surface: str = "toolsandbox"

    def __post_init__(self) -> None:
        self._base = ToolSandboxTypedSuffixOnlineAdapter(
            cheap_messages=self.cheap_messages,
            typed_row=self.typed_row,
            user_facing_message_required=True,
        )

    def cheap_conversation(self) -> Sequence[Mapping[str, Any]]:
        return self._base.cheap_conversation()

    def enumerate_frontier_candidates(
        self,
        prefix_payload: Mapping[str, Any],
    ) -> Sequence[CFRCOnlineCandidate]:
        return self._base.enumerate_frontier_candidates(prefix_payload)

    def build_frontier(
        self,
        candidate: CFRCOnlineCandidate,
        prefix_payload: Mapping[str, Any],
    ) -> Any:
        return self._base.build_frontier(candidate, prefix_payload)

    def bind_proposal(
        self,
        candidate: CFRCOnlineCandidate,
        proposal_graph: ProposalGraph,
        prefix_payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        # The grounded bridge already produced bindings that carry receipt
        # lineage, evidence refs, and replica status. Rebuilding them here would
        # drop exactly the fields the compiler checks.
        if self.grounded_bindings:
            return dict(self.grounded_bindings)
        return self._base.bind_proposal(candidate, proposal_graph, prefix_payload)

    def proposer_prompt_contract(
        self,
        candidate: CFRCOnlineCandidate,
        frontier: Any,
        prefix_payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return self._base.proposer_prompt_contract(candidate, frontier, prefix_payload)

    def execute_suffix(
        self,
        candidate: CFRCOnlineCandidate,
        compile_result: Any,
    ) -> CFRCOnlineExecution:
        payloads = _replay_payloads(
            copy.deepcopy(dict(payload))
            for payload in compile_result.suffix_payloads
            if isinstance(payload, Mapping)
        )
        reached_checkpoint = self.runtime.snapshot_context()
        reached_sha256 = self.runtime.context_sha256(reached_checkpoint)

        self.runtime.restore_context(reached_checkpoint)
        replica_result = self.runtime.execute_suffix(payloads)
        replica_post_sha256 = _context_replay_sha256(
            self.runtime,
            reached_checkpoint,
        )
        replica_error_count = _nonnegative_int(
            replica_result.get("runtime_error_count")
        )
        replica_clean = bool(
            replica_error_count == 0
            and replica_result.get("official_evaluator_used") is True
        )

        self.runtime.restore_context(reached_checkpoint)
        if self.runtime.context_sha256() != reached_sha256:
            raise RuntimeError("ToolSandbox reached-state restore mismatch")

        live_result: Mapping[str, Any] | None = None
        live_post_sha256: str | None = None
        if replica_clean:
            live_result = self.runtime.execute_suffix(payloads)
            live_post_sha256 = _context_replay_sha256(
                self.runtime,
                reached_checkpoint,
            )

        live_error_count = _nonnegative_int(
            live_result.get("runtime_error_count")
            if isinstance(live_result, Mapping)
            else None
        )
        replay_match = bool(
            replica_clean
            and isinstance(live_result, Mapping)
            and live_error_count == 0
            and live_result.get("official_evaluator_used") is True
            and live_post_sha256 == replica_post_sha256
        )
        if not replay_match:
            self.runtime.restore_context(reached_checkpoint)

        final_result = live_result if replay_match else self.runtime.evaluate()
        checked_error_count = max(
            replica_error_count,
            live_error_count,
            int(replica_clean and not replay_match),
        )
        replica_tool_calls = _nonnegative_int(
            replica_result.get("executed_tool_call_count")
        )
        live_tool_calls = _nonnegative_int(
            live_result.get("executed_tool_call_count")
            if isinstance(live_result, Mapping)
            else None
        )
        replica_assistant_responses = _nonnegative_int(
            replica_result.get("assistant_response_appended_count")
        )
        live_assistant_responses = _nonnegative_int(
            live_result.get("assistant_response_appended_count")
            if isinstance(live_result, Mapping)
            else None
        )
        summary = {
            "schema": "papc.toolsandbox.online_cfrc_controller.runtime_execution.v1",
            "provider_calls": 0,
            "replica_execution_performed": True,
            "private_reached_state_replica": True,
            "incremental_live_receipt_grounding": False,
            "live_execution_attempted": live_result is not None,
            "live_execution_committed": replay_match,
            "checked_execution_admitted": replay_match,
            "checked_execution_replay_match": replay_match,
            "reached_state_restored_on_block": not replay_match,
            "actual_tool_execution_performed": live_result is not None,
            "official_evaluator_used_at_runtime": bool(
                final_result.get("official_evaluator_used") is True
            ),
            "runtime_error_count": checked_error_count,
            "replica_executed_tool_call_count": replica_tool_calls,
            "live_executed_tool_call_count": live_tool_calls,
            "executed_tool_call_count": replica_tool_calls + live_tool_calls,
            "assistant_response_appended_count": (
                replica_assistant_responses + live_assistant_responses
            ),
            "tool_trace_recorded": bool(
                replay_match
                and replica_result.get("tool_trace_recorded") is True
                and isinstance(live_result, Mapping)
                and live_result.get("tool_trace_recorded") is True
            ),
            "residual_closure_evidence": replay_match,
            "evaluation": _public_evaluation(final_result),
        }
        assert_public_summary_safe(summary)
        return CFRCOnlineExecution(
            executed=replay_match,
            public_summary=summary,
        )


def _replay_payloads(
    payloads: Any,
) -> list[Mapping[str, Any]]:
    """Collapse grounded tool-call payloads into one deterministic code payload.

    Authoring runs each call on its own so the successor can read real receipts.
    Replaying them that way is not deterministic: every call lands its return
    value in a separate sandbox message, and a value read from the live clock
    differs between the replica pass and the live pass, so the two states never
    match and Checked Execution refuses a suffix that is in fact correct.

    The calls are already fixed by the time they are compiled, so they are
    replayed as one code payload in assignment form. Assignment keeps each
    return value in the tool trace, which the replay digest excludes, instead of
    in the message body, which it does not. The executed effects are unchanged.
    """

    lines: list[str] = []
    sequence: list[str] = []
    tail: list[Mapping[str, Any]] = []
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        calls = payload.get("tool_calls")
        if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
            calls = ()
        if payload.get("replica_succeeded") is False:
            # The compiler already accepted this residual, which means every
            # obligation is discharged by a call that succeeded. Replaying a call
            # that failed on the replica adds no effect and no evidence; it only
            # fails again and takes the whole suffix down with it.
            continue
        emitted = False
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            name = str(call.get("tool_name") or "")
            if not name.isidentifier():
                # Dropping the call would submit a suffix the compiler accepted
                # while silently omitting one of its actions, and the replay
                # digest would still match. Refuse instead.
                raise RuntimeError(
                    "compiler-selected tool call is not a callable identifier"
                )
            arguments = call.get("arguments")
            rendered = ", ".join(
                f"{key}={value!r}"
                for key, value in (
                    arguments.items() if isinstance(arguments, Mapping) else ()
                )
            )
            lines.append(f"_papc_r{len(lines)} = {name}({rendered})")
            sequence.append(name)
            emitted = True
        if not emitted:
            # Terminal message and declaration payloads carry no call to fold.
            tail.append(payload)
    replay: list[Mapping[str, Any]] = []
    if lines:
        replay.append(
            {
                "kind": "tool_call",
                "code": "\n".join(lines),
                "tool_sequence": sequence,
            }
        )
    replay.extend(tail)
    return replay


def _context_replay_sha256(
    runtime: ToolSandboxOnlineRuntime,
    reached_checkpoint: Any,
) -> str:
    replay_digest = getattr(runtime, "context_replay_sha256", None)
    if callable(replay_digest):
        return str(replay_digest(reached_checkpoint))
    return runtime.context_sha256()


def _strong_calls(result: CFRCOnlineResult) -> int:
    public = result.public_summary
    try:
        return max(0, int(public.get("strong_calls") or 0))
    except (TypeError, ValueError):
        return 0


def _frontier_suppressed(
    *,
    frontier_attempts: int,
    strong_call_count: int,
    config: ToolSandboxOnlineCFRCConfig,
) -> bool:
    """Report whether this episode may still open a frontier.

    Two independent limits apply: how many frontiers one episode is allowed, and
    how much of the shared provider-call budget is left.
    """

    return bool(
        frontier_attempts >= int(config.max_frontier_attempts)
        or strong_call_count >= int(config.max_strong_calls)
    )


def _execution_committed(result: CFRCOnlineResult) -> bool:
    return bool(
        result.decision == "execute"
        and result.execution is not None
        and result.execution.executed
    )


def _drained_strong_call(proposer: Any) -> RoleCall:
    drain = getattr(proposer, "drain_role_call", None)
    if callable(drain):
        return drain()
    return RoleCall(role="strong", token_usage_available=False)


# What the successor has to change, stated in terms of the residual rather than
# of the checker's internals. A reason code alone tends to produce the same
# proposal again.
_FEEDBACK_GUIDANCE: Mapping[str, str] = {
    "ungrounded_text_write_argument": (
        "A text argument of a state-changing call does not appear in the visible "
        "request. Copy the wording the user actually used."
    ),
    "ungrounded_numeric_write_argument": (
        "A numeric argument of a state-changing call came from neither the "
        "request nor an earlier result in this pass. Compute it first with the "
        "allowed read tools and pass that returned value."
    ),
    "unsupported_write_argument_type": (
        "A state-changing call passed a structured value. Pass scalars taken "
        "from the request or from an earlier result."
    ),
    "grounded_obligation_uncovered": (
        "An open obligation has no successful state-changing call. Complete it, "
        "or declare it with action=decline when the prefix cannot support it."
    ),
    "grounded_obligation_double_covered": (
        "An obligation was both completed and declined. Choose one."
    ),
    "terminal_response_missing": (
        "The pass ended without action=terminal. Finish with the final "
        "user-facing text once the residual is complete."
    ),
    "authoring_round_budget_exhausted": (
        "The step budget ran out before the terminal. Use fewer, more direct "
        "calls."
    ),
    "authoring_step_tool_outside_allowed_sequence": (
        "A call named a tool outside the authorized residual. Use only the "
        "listed tools, in order."
    ),
    "authoring_step_repeats_receipt": (
        "A call repeated an earlier call verbatim. Advance the residual instead."
    ),
    "binding_carrier_not_verified": (
        "A value was used without depending on the call that produced it. Read "
        "it from an earlier result in this pass."
    ),
    "node_carrier_not_verified": (
        "A value was used without depending on the call that produced it. Read "
        "it from an earlier result in this pass."
    ),
    "selected_effect_outside_frontier_allowed_units": (
        "A call changed state outside the authorized residual."
    ),
    "coverage_claim_replica_receipt_failed": (
        "The call meant to discharge an obligation failed on the replica. Fix "
        "its arguments and retry."
    ),
    "coverage_claim_missing_obligation": (
        "An open obligation has no action bound to it."
    ),
    "coverage_claim_missing_required_unit": (
        "An obligation was only partly discharged."
    ),
    "declaration_cannot_discharge_obligation": (
        "A declined obligation cannot complete the residual. Complete it with a "
        "real call if the prefix supports it."
    ),
    "declaration_not_accepted_by_contract": (
        "This contract cannot set an obligation aside. Discharge each one with a "
        "call that succeeds, and state anything you could not do in the terminal "
        "text instead."
    ),
}


def _typed_verifier_feedback(
    *,
    round_index: int,
    result: CFRCOnlineResult,
    diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    compile_result = result.compile_result
    compile_errors = (
        [str(error) for error in compile_result.compile_errors]
        if compile_result is not None
        else []
    )
    grounded_blockers = [
        str(item) for item in diagnostic.get("grounded_blockers") or ()
    ]
    authoring = diagnostic.get("authoring")
    authoring_blockers = [
        str(item)
        for item in (
            authoring.get("authoring_blockers")
            if isinstance(authoring, Mapping)
            else ()
        )
        or ()
    ]
    reasons = list(
        dict.fromkeys(compile_errors + grounded_blockers + authoring_blockers)
    )
    return {
        "round_index": int(round_index),
        "decision": result.decision,
        "refusal_reasons": reasons,
        "guidance": [
            _FEEDBACK_GUIDANCE[reason]
            for reason in reasons
            if reason in _FEEDBACK_GUIDANCE
        ],
    }


def _attempt_execution(attempt: _CFRCAttempt) -> Mapping[str, Any]:
    execution = attempt.public_summary.get("execution")
    return execution if isinstance(execution, Mapping) else {}


def _grounded_authoring_summary(
    attempts: Sequence[_CFRCAttempt],
) -> dict[str, Any]:
    """Aggregate authoring effort as counts only; arguments stay private."""

    authoring_rounds = 0
    replica_calls = 0
    failed_calls = 0
    declarations = 0
    reasons: list[str] = []
    terminal_with_last_call_enabled = False
    terminal_with_last_call_offered = 0
    terminal_with_last_call_accepted = 0
    terminal_with_last_call_rejection_reasons: list[str] = []
    for attempt in attempts:
        diagnostic = attempt.proposer_diagnostic
        authoring = diagnostic.get("authoring")
        if isinstance(authoring, Mapping):
            authoring_rounds += _nonnegative_int(authoring.get("authoring_round_count"))
            replica_calls += _nonnegative_int(
                authoring.get("replica_executed_call_count")
            )
            failed_calls += _nonnegative_int(
                authoring.get("replica_failed_call_count")
            )
            declarations += _nonnegative_int(authoring.get("declaration_count"))
            if authoring.get("terminal_with_last_call") is True:
                terminal_with_last_call_enabled = True
                terminal_with_last_call_offered += _nonnegative_int(
                    authoring.get("terminal_with_last_call_offered_count")
                )
                terminal_with_last_call_accepted += _nonnegative_int(
                    authoring.get("terminal_with_last_call_accepted_count")
                )
                terminal_with_last_call_rejection_reasons.extend(
                    str(item)
                    for item in authoring.get(
                        "terminal_with_last_call_rejection_reasons"
                    )
                    or ()
                )
            reasons.extend(
                str(item) for item in authoring.get("authoring_blockers") or ()
            )
        reasons.extend(str(item) for item in diagnostic.get("grounded_blockers") or ())
    summary = {
        "reproposal_round_count": len(attempts),
        "authoring_round_total": authoring_rounds,
        "replica_authoring_call_total": replica_calls,
        "replica_authoring_failed_call_total": failed_calls,
        "discharge_declaration_total": declarations,
        "grounded_refusal_reasons": sorted(set(reasons)),
    }
    if terminal_with_last_call_enabled:
        summary.update(
            {
                "terminal_with_last_call": True,
                "terminal_with_last_call_offered_total": (
                    terminal_with_last_call_offered
                ),
                "terminal_with_last_call_accepted_total": (
                    terminal_with_last_call_accepted
                ),
                "terminal_with_last_call_rejected_total": max(
                    0,
                    terminal_with_last_call_offered
                    - terminal_with_last_call_accepted,
                ),
                "terminal_with_last_call_rejection_reasons": sorted(
                    set(terminal_with_last_call_rejection_reasons)
                ),
            }
        )
    return summary


def _evaluation_payload(execution_summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "official_evaluator_used": bool(
            execution_summary.get("official_evaluator_used_at_runtime") is True
        ),
        "evaluation": dict(execution_summary.get("evaluation") or {}),
    }


def _public_evaluation(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, Mapping):
        evaluation = payload
    public: dict[str, Any] = {}
    for key, value in evaluation.items():
        if not isinstance(value, (int, float, bool)) or isinstance(value, bool):
            if isinstance(value, bool):
                public[str(key)] = value
            continue
        public[str(key)] = value
    return public


def _merged_runtime_ledger(
    runtime: ToolSandboxOnlineRuntime,
    cfrc_attempts: Sequence[_CFRCAttempt],
) -> dict[str, Any]:
    ledger = RuntimeLedger()
    runtime_summary_fn = getattr(runtime, "runtime_ledger_summary", None)
    if callable(runtime_summary_fn):
        runtime_summary = runtime_summary_fn()
        if isinstance(runtime_summary, Mapping):
            ledger.merge_public_summary(runtime_summary)
    for attempt in cfrc_attempts:
        attempt_ledger = attempt.public_summary.get("runtime_ledger")
        if isinstance(attempt_ledger, Mapping):
            ledger.merge_public_summary(attempt_ledger)
    return ledger.public_summary()


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


__all__ = [
    "SCHEMA",
    "PRIVATE_SCHEMA",
    "StrongProposer",
    "ToolSandboxFrontierMonitor",
    "ToolSandboxOnlineRuntime",
    "ToolSandboxOnlineCFRCConfig",
    "ToolSandboxOnlineCFRCController",
]

"""Native STATE-Bench and Agent-Diff loops for AgServe-QAC."""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from papc.baselines.agent_diff_live import (
    AgentDiffNativeAPI,
    AgentDiffNativeRunConfig,
    AgentDiffPrivateTask,
    _NativeSession,
    _parse_react_message,
)
from papc.baselines.agserve_qac import (
    AGSERVE_QAC_METHOD,
    AgServeArtifact,
    AgServeDecision,
    AgServeQACController,
    TransformerAgServeJudge,
)
from papc.baselines.router_accounting import CallReceipt
from papc.baselines.router_contract import CHEAP_MODEL, STRONG_MODEL, HistoryRouterView
from papc.baselines.state_bench_native_runtime import (
    StateBenchNativeAPI,
    StateBenchNativeRunConfig,
    StateBenchProviderBudget,
    StateBenchRuntimePolicy,
    _digest,
    _instrument_client,
    _usage_tuple,
    state_bench_public_history,
)
from papc.baselines.state_bench_router_driver import StateBenchOfficialTaskRef


@dataclass(frozen=True)
class AgServeStateBenchEpisode:
    official_score: float
    decisions: tuple[AgServeDecision, ...]
    provider_receipts: tuple[CallReceipt, ...]
    local_router_receipts: tuple[CallReceipt, ...]
    public_transcript: tuple[dict[str, Any], ...]
    case_root: Path

    def __post_init__(self) -> None:
        if self.official_score not in {0.0, 1.0}:
            raise ValueError("agserve_state_bench_score_not_binary")
        if not self.decisions:
            raise ValueError("agserve_state_bench_decisions_empty")
        if not self.provider_receipts:
            raise ValueError("agserve_state_bench_provider_receipts_empty")


@dataclass(frozen=True)
class AgServeAgentDiffEpisode:
    official_score: float
    decisions: tuple[AgServeDecision, ...]
    public_history: tuple[Mapping[str, Any], ...]
    provider_receipts: tuple[CallReceipt, ...]
    local_router_receipts: tuple[CallReceipt, ...]
    raw_path: Path

    def __post_init__(self) -> None:
        if self.official_score not in {0.0, 1.0}:
            raise ValueError("agserve_agent_diff_score_not_binary")
        if not self.decisions:
            raise ValueError("agserve_agent_diff_decisions_empty")
        if not self.provider_receipts:
            raise ValueError("agserve_agent_diff_provider_receipts_empty")


def run_agserve_state_bench_episode(
    *,
    task_ref: StateBenchOfficialTaskRef,
    api: StateBenchNativeAPI,
    config: StateBenchNativeRunConfig,
    policy: StateBenchRuntimePolicy,
    budget: StateBenchProviderBudget,
    judge: TransformerAgServeJudge,
    artifact: AgServeArtifact,
    registry_sha256: str,
    case_root: Path,
) -> AgServeStateBenchEpisode:
    """Run one source-bound STATE episode under the QAC turn policy."""

    if not isinstance(task_ref, StateBenchOfficialTaskRef):
        raise TypeError("agserve_state_bench_task_ref_invalid")
    if not isinstance(api, StateBenchNativeAPI):
        raise TypeError("agserve_state_bench_native_api_invalid")
    if not isinstance(config, StateBenchNativeRunConfig):
        raise TypeError("agserve_state_bench_config_invalid")
    if not isinstance(policy, StateBenchRuntimePolicy):
        raise TypeError("agserve_state_bench_policy_invalid")
    if not isinstance(budget, StateBenchProviderBudget):
        raise TypeError("agserve_state_bench_budget_invalid")
    if not isinstance(artifact, AgServeArtifact):
        raise TypeError("agserve_state_bench_artifact_invalid")
    if not _valid_sha256(registry_sha256):
        raise ValueError("agserve_state_bench_registry_sha256_invalid")

    task = _load_bound_state_task(api=api, task_ref=task_ref)
    domain = api.get_domain_config(task_ref.domain)
    policy.validate(api, require_embedding=False, require_automix=False)
    env_data, environment_path = api.load_task_environment(domain, task)
    if Path(environment_path).resolve() != task_ref.initial_environment_path:
        raise RuntimeError("agserve_state_bench_environment_path_mismatch")
    env = domain.environment_class(env_data.deep_copy(), now=task.now)
    root = Path(case_root).resolve()
    root.mkdir(parents=True, exist_ok=False)

    system_prompt = domain.agent_system_prompt.format(now=task.now, user_id=task.user_id)
    runtime_context = api.agent_runtime_context_class(
        task_id=task.task_id,
        user_id=task.user_id,
        domain=domain.name,
        now=task.now,
        task_summary=task.task_summary,
        state_requirements=task.state_requirements,
        task_requirements=task.task_requirements,
    )
    role_agents = {
        "cheap": api.state_bench_agent_class(
            policy.cheap_client,
            system_prompt,
            domain.tool_schemas,
            env.tool_handlers,
            runtime_context=runtime_context,
            agent_reasoning_effort=config.reasoning_effort,
        ),
        "strong": api.state_bench_agent_class(
            policy.strong_client,
            system_prompt,
            domain.tool_schemas,
            env.tool_handlers,
            runtime_context=runtime_context,
            agent_reasoning_effort=config.reasoning_effort,
        ),
    }
    controller = AgServeQACController(
        judge=judge,
        qj_threshold=artifact.qj_threshold,
        rj_threshold=artifact.rj_threshold,
        check_frequency=artifact.state_check_frequency,
    )
    decisions: list[AgServeDecision] = []
    local_receipts: list[CallReceipt] = []
    base_agent = api.base_agent_class
    turn_cap = int(domain.max_agent_turns)

    class AgServeStateAgent(base_agent):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__(runtime_context=runtime_context)

        def act(
            self,
            conversation: list[Any],
        ) -> tuple[str, list[dict[str, Any]], list[Any]]:
            public_history = state_bench_public_history(conversation)
            view = HistoryRouterView(
                original_request=task.opening_message,
                public_history=public_history,
                turn_index=len(decisions),
                turn_cap=turn_cap,
            )
            started = time.perf_counter()
            decision = controller.decide(view)
            latency = max(0.0, time.perf_counter() - started)
            decisions.append(decision)
            local_receipts.append(
                CallReceipt.local_router(
                    method=AGSERVE_QAC_METHOD,
                    latency_seconds=latency,
                )
            )
            selected = role_agents[decision.selected_role]
            before_usage = _usage_tuple(selected.token_usage)
            response = selected.act(conversation)
            after_usage = _usage_tuple(selected.token_usage)
            self.add_token_usage(
                input_tokens=after_usage[0] - before_usage[0],
                cached_input_tokens=after_usage[1] - before_usage[1],
                output_tokens=after_usage[2] - before_usage[2],
                reasoning_output_tokens=after_usage[3] - before_usage[3],
            )
            return response

    before_receipts = len(budget.receipts)
    with ExitStack() as stack:
        for client, role, model, effort, phase in (
            (
                policy.cheap_client,
                "cheap_agent",
                CHEAP_MODEL,
                config.reasoning_effort,
                "native_agent",
            ),
            (
                policy.strong_client,
                "strong_agent",
                STRONG_MODEL,
                config.reasoning_effort,
                "native_agent",
            ),
            (
                policy.simulator_client,
                "user_simulator",
                config.user_model,
                "protocol",
                "native_user_simulator",
            ),
        ):
            stack.enter_context(
                _instrument_client(
                    client=client,
                    budget=budget,
                    role=role,
                    model=model,
                    reasoning_effort=effort,
                    phase=phase,
                )
            )
        trajectory = api.run_task(
            task=task,
            env_data=env_data,
            user_id=task.user_id,
            client=None,
            domain=domain,
            agent=AgServeStateAgent(),
            env=env,
            simulator_client=policy.simulator_client,
            trajectory_metadata={
                "router_method": AGSERVE_QAC_METHOD,
                "execution_level": "turn",
                "registry_sha256": registry_sha256,
                "agserve_scope": "quality_aware_client_only",
            },
        )
    trajectory.save(root / "trajectory_generated.json")
    unscored_path = root / "trajectory_unscored.json"
    scored_path = root / "trajectory_scored.json"
    trajectory.save(unscored_path)
    with _instrument_client(
        client=policy.evaluator_client,
        budget=budget,
        role="evaluator",
        model=config.evaluator_model,
        reasoning_effort=config.reasoning_effort,
        phase="official_score",
    ):
        score_result = api.score_one(
            unscored_path,
            task_ref.native_task_path.parent,
            policy.task_requirements_judge,
            policy.ux_judge,
            scored_path,
            api.protocol,
            task_ref.domain,
        )
    if score_result.get("status") != "OK":
        raise RuntimeError("agserve_state_bench_official_scorer_failed")
    scored = json.loads(scored_path.read_text(encoding="utf-8"))
    official_score = scored.get("task_completion_pass")
    if official_score not in {0, 0.0, 1, 1.0}:
        raise TypeError("agserve_state_bench_official_score_invalid")
    return AgServeStateBenchEpisode(
        official_score=float(official_score),
        decisions=tuple(decisions),
        provider_receipts=tuple(budget.receipts[before_receipts:]),
        local_router_receipts=tuple(local_receipts),
        public_transcript=state_bench_public_history(trajectory.conversation),
        case_root=root,
    )


def run_agserve_agent_diff_episode(
    *,
    task: AgentDiffPrivateTask,
    api: AgentDiffNativeAPI,
    config: AgentDiffNativeRunConfig,
    provider_client: Any,
    broker: Any,
    raw_path: Path,
    api_key_env: str,
    judge: TransformerAgServeJudge,
    artifact: AgServeArtifact,
) -> AgServeAgentDiffEpisode:
    """Run one frozen Agent-Diff episode with QAC state-preserving upgrade."""

    if not isinstance(artifact, AgServeArtifact):
        raise TypeError("agserve_agent_diff_artifact_invalid")
    controller = AgServeQACController(
        judge=judge,
        qj_threshold=artifact.qj_threshold,
        rj_threshold=artifact.rj_threshold,
        check_frequency=artifact.agent_diff_check_frequency,
    )
    session = _NativeSession(
        task=task,
        api=api,
        config=config,
        provider_client=provider_client,
        broker=broker,
        raw_path=Path(raw_path).resolve(),
        api_key_env=api_key_env,
    )
    history: list[dict[str, Any]] = [
        {"role": "user", "content": task.ref.original_request}
    ]
    decisions: list[AgServeDecision] = []
    local_receipts: list[CallReceipt] = []
    error: BaseException | None = None
    try:
        session.start()
        for iteration in range(1, config.max_iterations + 1):
            view = HistoryRouterView(
                original_request=task.ref.original_request,
                public_history=tuple(history),
                turn_index=iteration - 1,
                turn_cap=config.max_iterations,
            )
            started = time.perf_counter()
            decision = controller.decide(view)
            latency = max(0.0, time.perf_counter() - started)
            decisions.append(decision)
            local_receipts.append(
                CallReceipt.local_router(
                    method=AGSERVE_QAC_METHOD,
                    latency_seconds=latency,
                )
            )
            text, receipt = session.model_turn(
                model=decision.selected_model,
                public_history=history,
            )
            parsed = _parse_react_message(text)
            event: dict[str, Any] = {
                "iteration": iteration,
                "model": decision.selected_model,
                "selected_role": decision.selected_role,
                "agserve_qac_decision": decision.public_record(),
                "local_judge_latency_seconds": latency,
                "assistant_text": text,
                "assistant_text_sha256": hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest(),
                "receipt": receipt.public_record(),
            }
            action = parsed["action"]
            done = parsed["done"]
            if action:
                tool = session.execute_action(action)
                event.update(tool)
                session.trace.append(event)
                history.extend(
                    (
                        {"role": "assistant", "content": text},
                        {
                            "role": "user",
                            "content": (
                                "<observation>\n"
                                f"{tool['observation']}\n"
                                "</observation>"
                            ),
                        },
                    )
                )
                continue
            if done:
                event["done"] = done
                session.trace.append(event)
                history.append({"role": "assistant", "content": text})
                session.completion_reason = "done_tag"
                break
            event["format_error"] = "missing_action_or_done"
            session.trace.append(event)
            history.extend(
                (
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": (
                            "missing_action_or_done: respond using either "
                            "<action>one bash command</action> or "
                            "<done>brief summary</done>."
                        ),
                    },
                )
            )
        else:
            session.completion_reason = "max_iterations_exhausted"
        evaluation = session.evaluate()
        score = 1.0 if evaluation["passed"] is True else 0.0
        return AgServeAgentDiffEpisode(
            official_score=score,
            decisions=tuple(decisions),
            public_history=tuple(history),
            provider_receipts=tuple(session.receipts),
            local_router_receipts=tuple(local_receipts),
            raw_path=Path(raw_path).resolve(),
        )
    except BaseException as exc:
        error = exc
        raise
    finally:
        session.finalize(error=error)


def _load_bound_state_task(
    *,
    api: StateBenchNativeAPI,
    task_ref: StateBenchOfficialTaskRef,
) -> Any:
    task = api.task_definition_class.load(task_ref.native_task_path)
    if str(task.task_id) != task_ref.native_task_locator:
        raise RuntimeError("agserve_state_bench_task_locator_mismatch")
    task_payload = json.loads(task_ref.native_task_path.read_text(encoding="utf-8"))
    environment_payload = json.loads(
        task_ref.initial_environment_path.read_text(encoding="utf-8")
    )
    if _digest(environment_payload) != task_ref.initial_environment_sha256:
        raise RuntimeError("agserve_state_bench_environment_hash_mismatch")
    if (
        _digest({"task": task_payload, "initial_environment": environment_payload})
        != task_ref.payload_sha256
    ):
        raise RuntimeError("agserve_state_bench_task_payload_hash_mismatch")
    return task


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "AgServeAgentDiffEpisode",
    "AgServeStateBenchEpisode",
    "run_agserve_agent_diff_episode",
    "run_agserve_state_bench_episode",
]

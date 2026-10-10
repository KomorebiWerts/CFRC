"""StepWise on the source-bound STATE-Bench native turn interface."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from papc.baselines.router_accounting import CallReceipt
from papc.baselines.router_contract import (
    CHEAP_MODEL,
    STRONG_MODEL,
    HistoryRouterView,
)
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
from papc.baselines.stepwise_agent_diff_runtime import _milestone_verification
from papc.baselines.stepwise_upstream import (
    STEPWISE_METHOD,
    StepWiseController,
    StepWiseDecision,
    StepWiseMonitor,
)


@dataclass(frozen=True)
class StepWiseStateBenchEpisode:
    official_score: float
    decisions: tuple[StepWiseDecision, ...]
    provider_receipts: tuple[CallReceipt, ...]
    local_router_receipts: tuple[CallReceipt, ...]
    public_transcript: tuple[dict[str, Any], ...]
    case_root: Path

    def __post_init__(self) -> None:
        if self.official_score not in {0.0, 1.0}:
            raise ValueError("stepwise_state_bench_score_not_binary")
        if not self.decisions:
            raise ValueError("stepwise_state_bench_decisions_empty")
        if not self.provider_receipts:
            raise ValueError("stepwise_state_bench_provider_receipts_empty")


class StateBenchVerifierBroker:
    """Charge one StepWise verifier call to the shared native budget."""

    def __init__(self, budget: StateBenchProviderBudget) -> None:
        self.budget = budget

    def perform(self, raw_attempt: Callable[[], CallReceipt]) -> CallReceipt:
        self.budget.reserve()
        receipt = raw_attempt()
        self.budget.settle(receipt)
        return receipt


def run_stepwise_state_bench_episode(
    *,
    task_ref: StateBenchOfficialTaskRef,
    api: StateBenchNativeAPI,
    config: StateBenchNativeRunConfig,
    policy: StateBenchRuntimePolicy,
    budget: StateBenchProviderBudget,
    monitor: StepWiseMonitor,
    verifier_provider_client: Any,
    registry_sha256: str,
    case_root: Path,
) -> StepWiseStateBenchEpisode:
    """Run cheap-first StepWise from one native STATE initial environment."""

    if not isinstance(task_ref, StateBenchOfficialTaskRef):
        raise TypeError("stepwise_state_bench_task_ref_invalid")
    if not isinstance(api, StateBenchNativeAPI):
        raise TypeError("stepwise_state_bench_native_api_invalid")
    if not isinstance(config, StateBenchNativeRunConfig):
        raise TypeError("stepwise_state_bench_config_invalid")
    if not isinstance(policy, StateBenchRuntimePolicy):
        raise TypeError("stepwise_state_bench_policy_invalid")
    if not isinstance(budget, StateBenchProviderBudget):
        raise TypeError("stepwise_state_bench_budget_invalid")
    if not callable(monitor):
        raise TypeError("stepwise_state_bench_monitor_invalid")
    if (
        not isinstance(registry_sha256, str)
        or len(registry_sha256) != 64
        or any(character not in "0123456789abcdef" for character in registry_sha256)
    ):
        raise ValueError("stepwise_state_bench_registry_sha256_invalid")

    task = _load_bound_task(api=api, task_ref=task_ref)
    domain = api.get_domain_config(task_ref.domain)
    policy.validate(api, require_embedding=False, require_automix=False)
    env_data, environment_path = api.load_task_environment(domain, task)
    if Path(environment_path).resolve() != task_ref.initial_environment_path:
        raise RuntimeError("stepwise_state_bench_environment_path_mismatch")
    env = domain.environment_class(env_data.deep_copy(), now=task.now)
    root = Path(case_root).resolve()
    root.mkdir(parents=True, exist_ok=False)

    system_prompt = domain.agent_system_prompt.format(
        now=task.now,
        user_id=task.user_id,
    )
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
    controller = StepWiseController()
    decisions: list[StepWiseDecision] = []
    local_receipts: list[CallReceipt] = []
    verifier_broker = StateBenchVerifierBroker(budget)
    base_agent = api.base_agent_class
    turn_cap = int(domain.max_agent_turns)

    class StepWiseStateAgent(base_agent):  # type: ignore[misc, valid-type]
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
            monitor_latency = 0.0

            def timed_monitor(
                stuck_text: str,
                milestone_text: str,
            ) -> tuple[float, float]:
                nonlocal monitor_latency
                started = time.perf_counter()
                result = monitor(stuck_text, milestone_text)
                monitor_latency += max(0.0, time.perf_counter() - started)
                return result

            def verify(packet: str) -> bool:
                _, verified = _milestone_verification(
                    packet=packet,
                    provider_client=verifier_provider_client,
                    broker=verifier_broker,
                )
                return verified

            decision = controller.decide(
                view=view,
                monitor=timed_monitor,
                verifier=verify,
            )
            decisions.append(decision)
            local_receipts.append(
                CallReceipt.local_router(
                    method=STEPWISE_METHOD,
                    latency_seconds=monitor_latency,
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
            agent=StepWiseStateAgent(),
            env=env,
            simulator_client=policy.simulator_client,
            trajectory_metadata={
                "router_method": STEPWISE_METHOD,
                "execution_level": "turn",
                "registry_sha256": registry_sha256,
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
        raise RuntimeError("stepwise_state_bench_official_scorer_failed")
    scored = json.loads(scored_path.read_text(encoding="utf-8"))
    official_score = scored.get("task_completion_pass")
    if official_score not in {0, 0.0, 1, 1.0}:
        raise TypeError("stepwise_state_bench_official_score_invalid")
    return StepWiseStateBenchEpisode(
        official_score=float(official_score),
        decisions=tuple(decisions),
        provider_receipts=tuple(budget.receipts[before_receipts:]),
        local_router_receipts=tuple(local_receipts),
        public_transcript=state_bench_public_history(trajectory.conversation),
        case_root=root,
    )


def _load_bound_task(
    *,
    api: StateBenchNativeAPI,
    task_ref: StateBenchOfficialTaskRef,
) -> Any:
    task = api.task_definition_class.load(task_ref.native_task_path)
    if str(task.task_id) != task_ref.native_task_locator:
        raise RuntimeError("stepwise_state_bench_task_locator_mismatch")
    task_payload = json.loads(task_ref.native_task_path.read_text(encoding="utf-8"))
    environment_payload = json.loads(
        task_ref.initial_environment_path.read_text(encoding="utf-8")
    )
    if _digest(environment_payload) != task_ref.initial_environment_sha256:
        raise RuntimeError("stepwise_state_bench_environment_hash_mismatch")
    if (
        _digest(
            {
                "task": task_payload,
                "initial_environment": environment_payload,
            }
        )
        != task_ref.payload_sha256
    ):
        raise RuntimeError("stepwise_state_bench_task_payload_hash_mismatch")
    return task


__all__ = [
    "StateBenchVerifierBroker",
    "StepWiseStateBenchEpisode",
    "run_stepwise_state_bench_episode",
]

"""Native Tau2 agent adapter for AgServe-QAC."""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any

from papc.baselines.agserve_qac import (
    AGSERVE_QAC_METHOD,
    AgServeArtifact,
    AgServeDecision,
    AgServeQACController,
    TransformerAgServeJudge,
)
from papc.baselines.router_accounting import CallReceipt
from papc.baselines.router_contract import REASONING_EFFORT, HistoryRouterView


def register_agserve_tau2_agent(
    *,
    api: Any,
    agent_name: str,
    judge: TransformerAgServeJudge,
    artifact: AgServeArtifact,
    generate_callback: Any,
    decisions: list[AgServeDecision],
    local_receipts: list[CallReceipt],
    api_base: str,
    max_steps: int,
) -> str:
    """Register AgServe's public-history QAC policy in native Tau2."""

    from tau2.data_model.message import MultiToolMessage, UserMessage

    from papc.baselines.tau2_native_runtime import tau2_public_history

    frequencies = artifact.metadata.get("check_frequencies", {})
    retail_check_frequency = frequencies.get("tau2_retail")
    if (
        isinstance(retail_check_frequency, bool)
        or not isinstance(retail_check_frequency, int)
        or retail_check_frequency <= 0
    ):
        raise ValueError("agserve_tau2_retail_frequency_missing")
    if not callable(generate_callback):
        raise TypeError("agserve_tau2_generate_callback_invalid")
    if not isinstance(agent_name, str) or not agent_name.strip():
        raise ValueError("agserve_tau2_agent_name_invalid")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
        raise ValueError("agserve_tau2_max_steps_invalid")

    base = api.llm_agent_class
    controller = AgServeQACController(
        judge=judge,
        qj_threshold=artifact.qj_threshold,
        rj_threshold=artifact.rj_threshold,
        check_frequency=retail_check_frequency,
    )

    class AgServeTau2Agent(base):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._papc_seed: int | None = None
            self._papc_original_request: str | None = None

        def set_seed(self, seed: int) -> None:
            self._papc_seed = seed
            super().set_seed(seed)

        def _generate_next_message(self, message: Any, state: Any) -> Any:
            if isinstance(message, UserMessage) and message.is_audio:
                raise ValueError("agserve_tau2_audio_user_unsupported")
            if isinstance(message, MultiToolMessage):
                state.messages.extend(message.tool_messages)
            else:
                state.messages.append(message)
            public_history = tau2_public_history(tuple(state.messages))
            if self._papc_original_request is None:
                self._papc_original_request = next(
                    str(item.get("content", ""))
                    for item in public_history
                    if item.get("role") == "user" and item.get("content")
                )
            view = HistoryRouterView(
                original_request=self._papc_original_request,
                public_history=public_history,
                turn_index=len(decisions),
                turn_cap=max_steps,
            )
            started = time.perf_counter()
            decision = controller.decide(view)
            local_receipts.append(
                CallReceipt.local_router(
                    method=AGSERVE_QAC_METHOD,
                    latency_seconds=max(0.0, time.perf_counter() - started),
                )
            )
            decisions.append(decision)
            self.llm = decision.selected_model
            llm_args = {
                "api_base": api_base,
                "reasoning_effort": REASONING_EFFORT,
            }
            if self._papc_seed is not None:
                llm_args["seed"] = self._papc_seed
            self.llm_args = deepcopy(llm_args)
            return generate_callback(
                model=self.llm,
                tools=self.tools,
                messages=state.system_messages + state.messages,
                call_name="agent_response",
                **self.llm_args,
            )

    def factory(
        tools: Any,
        domain_policy: str,
        llm: str,
        llm_args: Any = None,
        **_: Any,
    ) -> Any:
        return AgServeTau2Agent(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=dict(llm_args or {}),
        )

    if api.registry.get_agent_factory(agent_name) is not None:
        raise ValueError("agserve_tau2_agent_name_already_registered")
    api.registry.register_agent_factory(factory, agent_name)
    return agent_name


__all__ = ["register_agserve_tau2_agent"]

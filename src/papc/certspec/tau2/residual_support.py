"""Small runtime helpers for the active tau2 shadow-graph path.

The historical Retail runner contains experiment construction and closed
adapters. The active Airline and Retail methods only need compact
visible evidence, append-only JSONL, result accounting, and launch helpers.
Keeping those here prevents the live method entrypoint from importing the
legacy controller.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from papc.certspec.cfrc_compiler import CFRCCompileResult
from papc.certspec.costing import (
    OPENAI_PUBLIC_PRICING_SNAPSHOT_DATE,
    OPENAI_PUBLIC_PRICING_SOURCE,
    cache_neutral_cost_usd,
)
from papc.certspec.grounded_proposal import GroundedProposalBundle, receipt_ref
from papc.certspec.provider import UsageAccumulator
from papc.certspec.runtime_ledger import RuntimeLedger


@dataclass(frozen=True)
class CompiledGraphStep:
    """One compiler-selected node in dependency order.

    ``payload`` remains private. The public compiler result exposes only its
    shape, while Checked Execution consumes this exact sequence.
    """

    node_id: str
    kind: str
    dependency_ids: tuple[str, ...]
    payload: Mapping[str, Any]

    @property
    def emits_live_action(self) -> bool:
        """Whether Checked Execution releases anything for this node.

        A discharge declaration records a requested failure return. It cannot
        close an obligation and releases no live tool call or standalone message.
        """

        return str(self.payload.get("kind") or "") != "declaration"


def compiled_graph_steps(
    *,
    bundle: GroundedProposalBundle,
    compiler_result: CFRCCompileResult,
) -> tuple[CompiledGraphStep, ...]:
    """Materialize the whole compiler-selected graph as the execution queue."""

    if compiler_result.decision != "execute":
        return ()
    by_id = bundle.proposal_graph.node_by_id()
    selected = tuple(compiler_result.selected_node_ids)
    if set(selected) != set(by_id) or len(selected) != len(by_id):
        raise ValueError("checked execution requires whole-graph selection")

    emitted: set[str] = set()
    steps: list[CompiledGraphStep] = []
    for node_id in selected:
        node = by_id[node_id]
        if not set(node.dependency_ids) <= emitted:
            raise ValueError("compiler result is not in graph dependency order")
        binding = bundle.bindings.get(node_id)
        if binding is None or not isinstance(binding.suffix_payload, Mapping):
            raise ValueError("compiler-selected graph node lacks a private payload")
        steps.append(
            CompiledGraphStep(
                node_id=node_id,
                kind=node.kind,
                dependency_ids=tuple(node.dependency_ids),
                payload=dict(binding.suffix_payload),
            )
        )
        emitted.add(node_id)
    return tuple(steps)


def graph_step_receipt_matches(step: CompiledGraphStep, incoming: Any) -> bool:
    """Bind one live tau2 receipt to the selected graph node's replica receipt."""

    if step.kind != "tool_call":
        return False
    rows = _tool_messages(incoming)
    if len(rows) != 1:
        return False
    receipt = rows[0]
    if bool(_message_attr(receipt, "error")):
        return False
    expected_call_id = str(step.payload.get("tool_call_id") or "")
    if expected_call_id and _message_attr_str(receipt, "id") != expected_call_id:
        return False

    tool_calls = step.payload.get("tool_calls")
    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        return False
    call = tool_calls[0]
    if not isinstance(call, Mapping):
        return False
    tool_name = str(call.get("tool_name") or "")
    arguments = call.get("arguments")
    if not tool_name or not isinstance(arguments, Mapping):
        return False
    actual_ref = receipt_ref(
        tool_name=tool_name,
        arguments=dict(arguments),
        result=_receipt_content_value(_message_attr(receipt, "content")),
        instance_id=str(step.payload.get("receipt_instance_id") or step.node_id),
    )
    return actual_ref == str(step.payload.get("expected_receipt_ref") or "")


def execute_on_replica(
    *,
    call: Any,
    tool: Any,
    tool_message_type: Any,
) -> tuple[Any, Any]:
    """Execute one native call on a private replica and return its safe receipt."""

    error = False
    try:
        value = tool(**dict(_tool_call_prompt_json(call)["arguments"]))
        safe_value = json_safe_value(value)
    except Exception as exc:
        safe_value = f"Error: {exc}"
        error = True
    receipt = tool_message_type(
        id=_message_attr_str(call, "id"),
        role="tool",
        content=(
            safe_value
            if isinstance(safe_value, str)
            else json.dumps(safe_value, default=str)
        ),
        requestor=_message_attr_str(call, "requestor") or "assistant",
        error=error,
    )
    return receipt, safe_value


def runtime_ledger_for_messages(
    *,
    cheap_messages: Sequence[Any] = (),
    strong_messages: Sequence[Any] = (),
) -> dict[str, Any]:
    ledger = RuntimeLedger()
    for role, messages in (
        ("cheap", cheap_messages),
        ("strong", strong_messages),
    ):
        for message in messages:
            ledger.add_usage(
                role,
                usage_with_raw_cache_details(message),
                latency_seconds=_optional_float(
                    _message_attr(message, "generation_time_seconds")
                ),
                cost_usd=_optional_float(_message_attr(message, "cost")),
            )
    return ledger.public_summary()


def usage_with_raw_cache_details(message: Any) -> dict[str, Any] | None:
    """Recover cache metadata that upstream tau2 omits from Message.usage."""

    usage = _mapping_or_none(_message_attr(message, "usage"))
    raw_data = _mapping_or_none(_message_attr(message, "raw_data")) or {}
    raw_usage = _mapping_or_none(raw_data.get("usage"))
    if usage is None:
        return dict(raw_usage) if raw_usage is not None else None

    merged = dict(usage)
    if raw_usage is not None:
        # tau2's Message.usage keeps only the two flat token counts, so the
        # output-side detail blocks have to be recovered here as well. Without
        # them every tau2 artifact recorded zero reasoning tokens, including for
        # models that report reasoning outside their completion count.
        for key in (
            "cached_input_tokens",
            "input_tokens_details",
            "prompt_tokens_details",
            "output_tokens_details",
            "completion_tokens_details",
            "reasoning_tokens",
            "reasoning_output_tokens",
        ):
            if merged.get(key) is None and raw_usage.get(key) is not None:
                merged[key] = raw_usage[key]
    return merged


def json_safe_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return json_safe_value(value.model_dump())
    if isinstance(value, Mapping):
        return {str(key): json_safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def attach_metadata(message: Any, metadata: Mapping[str, Any]) -> Any:
    if isinstance(message, dict):
        raw_data = message.get("raw_data")
        if not isinstance(raw_data, dict):
            raw_data = {}
            message["raw_data"] = raw_data
        raw_data["papc_cfrc"] = dict(metadata)
        return message
    raw_data = getattr(message, "raw_data", None)
    if not isinstance(raw_data, dict):
        raw_data = {}
        setattr(message, "raw_data", raw_data)
    raw_data["papc_cfrc"] = dict(metadata)
    return message


def write_jsonl_append(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def compact_visible_evidence(
    messages: Sequence[Any],
    *,
    max_events: int = 16,
    max_text_chars: int = 800,
    include_assistant_text: bool = False,
    exclude_tool_receipt_ids: frozenset[str] = frozenset(),
    user_requests_carried_verbatim: bool = False,
) -> list[dict[str, Any]]:
    """Serialize visible reached-state events without provider metadata.

    ``exclude_tool_receipt_ids`` names receipts that already reach the model
    through another carrier, so this packet keeps only a stub for them instead
    of paying for the same payload twice in every turn's input.

    ``user_requests_carried_verbatim`` says the same about the user turns, and
    it is off by default because it is only true of a caller that replays the
    whole transcript beside this packet. Airline does; Retail does not, and
    hands the successor nothing but this packet, so there the request text is
    the only copy there is.
    """

    rows: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        role = _message_attr_str(message, "role")
        row: dict[str, Any] = {"event_id": f"e{index}", "role": role}
        content = _message_attr(message, "content")
        tool_calls = _message_tool_calls(message)
        if role == "tool" and exclude_tool_receipt_ids:
            receipt_id = _message_attr_str(message, "id") or _message_attr_str(
                message, "tool_call_id"
            )
            if receipt_id and receipt_id in exclude_tool_receipt_ids:
                row["kind"] = "visible_tool_result"
                row["tool_call_id"] = receipt_id
                row["carried_verbatim_in_replica_transcript"] = True
                rows.append(row)
                continue
        if tool_calls:
            carried_call_ids = [
                _message_attr_str(call, "id") for call in tool_calls
            ]
            if exclude_tool_receipt_ids and all(
                call_id and call_id in exclude_tool_receipt_ids
                for call_id in carried_call_ids
            ):
                # The message that authored these calls is replayed verbatim
                # next to their receipts, so spelling its arguments again here
                # bills the same carrier twice on every turn.
                row["kind"] = "visible_tool_call_carrier"
                row["tool_call_ids"] = carried_call_ids
                row["carried_verbatim_in_replica_transcript"] = True
                rows.append(row)
                continue
            row["kind"] = "visible_tool_call_carrier"
            row["tool_calls"] = [_tool_call_prompt_json(call) for call in tool_calls]
        elif role == "user" and content is not None:
            row["kind"] = "visible_user_request"
            if user_requests_carried_verbatim:
                # The transcript beside this packet already carries the turn in
                # full, so the truncated copy here bills the same words twice
                # on every round. The stub keeps the event in the packet's
                # ordering, which is what the rest of the packet indexes into.
                row["carried_verbatim_in_replica_transcript"] = True
            else:
                row["content"] = _truncate_text(str(content), max_text_chars)
        elif (
            include_assistant_text
            and role == "assistant"
            and content is not None
            and str(content).strip()
        ):
            # Retain assistant proposals only as untrusted confirmation
            # context. They may help resolve a short approval utterance, but
            # they cannot become authority evidence.
            row["kind"] = "untrusted_confirmation_context"
            row["authority"] = False
            row["content"] = _truncate_text(str(content), max_text_chars)
        elif role == "tool" or _message_attr_str(message, "tool_call_id"):
            row["kind"] = "visible_tool_result"
            tool_name = _message_attr_str(message, "name") or _message_attr_str(
                message, "tool_name"
            )
            if tool_name:
                row["tool_name"] = tool_name
            tool_call_id = _message_attr_str(message, "tool_call_id")
            if tool_call_id:
                row["tool_call_id"] = tool_call_id
            if content is not None:
                row["content"] = _truncate_text(str(content), max_text_chars)
        else:
            continue
        rows.append(row)
    return rows[-max_events:]


def safe_agent_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value)
    return cleaned[:96] or "papc_tau2_shadow_graph_agent"


def llm_args(api_base: str, reasoning_effort: str) -> dict[str, Any]:
    args: dict[str, Any] = {
        "temperature": 0.0,
        "drop_params": True,
        "reasoning_effort": reasoning_effort,
    }
    if api_base:
        args["api_base"] = api_base
    return args


def llm_args_for_model(
    api_base: str, reasoning_effort: str, model: str
) -> dict[str, Any]:
    """Bind gateway routing and omit the Gemini bridge's unsupported knob.

    CLIProxy's OpenAI-compatible Gemini route can return an empty assistant
    message when ``reasoning_effort`` is sent alongside a multi-turn tool
    transcript.  Keep the registered effort in the launch metadata, but omit
    the transport parameter for Gemini requests. GPT gateway routing must not
    depend on whether LiteLLM's optional pricing catalog knows a newer model.
    """

    args = llm_args(api_base, reasoning_effort)
    normalized = str(model or "").strip().lower()
    if normalized.startswith("openai/"):
        normalized = normalized[len("openai/") :]
    if normalized.startswith("gemini-"):
        args.pop("reasoning_effort", None)
    if api_base and normalized.startswith("gpt-"):
        args["custom_llm_provider"] = "openai"
    return args


def litellm_model_name(model: str) -> str:
    """Return the route spelling used by the shared OpenAI-compatible client.

    LiteLLM treats a bare ``claude-*`` or ``gemini-*`` name as a native provider
    route.  The active tau2 runners use the local OpenAI-compatible gateway for
    both families, where the ``openai/`` prefix selects that route and is
    removed before the request reaches the gateway.  Normalize it once at the
    runner boundary so a caller cannot silently switch provider paths just by
    omitting the transport prefix.
    """

    value = str(model or "").strip()
    if value.lower().startswith("openai/"):
        return value
    if value.lower().startswith(("gemini-", "claude-")):
        return f"openai/{value}"
    return value


def retimestamp_delayed_message(message: Any) -> Any:
    """Stamp a proposal-time message when Checked Execution releases it.

    tau2 sorts the finalized trajectory by message timestamp before replaying
    it for official evaluation. A later graph node or terminal message was
    authored on the private replica before its live receipt existed, so its
    original timestamp would otherwise move it ahead of that receipt.
    """

    timestamp = datetime.now().isoformat()
    if isinstance(message, dict):
        message["timestamp"] = timestamp
    else:
        setattr(message, "timestamp", timestamp)
    return message


def arm_public_summary(
    *,
    arm_name: str,
    results_payload: Mapping[str, Any],
    sidecar_rows: Sequence[Mapping[str, Any]],
    cheap_model: str,
    strong_model: str,
) -> dict[str, Any]:
    simulations = [
        row
        for row in _list_value(results_payload.get("simulations"))
        if isinstance(row, Mapping)
    ]
    evaluable = [
        row
        for row in simulations
        if isinstance(row.get("reward_info"), Mapping)
        and row["reward_info"].get("reward") is not None
    ]
    rewards = [float(row["reward_info"]["reward"]) for row in evaluable]
    ledger = tau2_runtime_ledger_from_results(
        results_payload=results_payload,
        cheap_model=cheap_model,
        strong_model=strong_model,
        sidecar_rows=sidecar_rows,
    )
    provider_reported_cost = _provider_reported_agent_side_cost(ledger)
    cache_neutral_cost = _cache_neutral_agent_side_cost(
        ledger,
        cheap_model=cheap_model,
        strong_model=strong_model,
    )
    return {
        "schema": "papc.tau2.shadow_graph_arm_public_summary.v1",
        "arm": arm_name,
        "task_count": len(simulations),
        "reward_evaluable_count": len(evaluable),
        "infrastructure_excluded_count": len(simulations) - len(evaluable),
        "reward_sum": sum(rewards) if simulations else None,
        "reward_mean": (sum(rewards) / len(rewards)) if rewards else None,
        "official_action_closure": _official_action_closure(evaluable),
        "episode_wall_clock_seconds": _episode_wall_clock_summary(simulations),
        "runtime_ledger": ledger,
        "runtime_ledger_completeness": _runtime_ledger_completeness(ledger),
        "runtime_ledger_token_reconciliation": _ledger_token_reconciliation(
            results_payload=results_payload,
            sidecar_rows=sidecar_rows,
            ledger=ledger,
        ),
        "agent_side_raw_cost_usd": (
            cache_neutral_cost
            if cache_neutral_cost is not None
            else provider_reported_cost
        ),
        "agent_side_provider_reported_cost_usd": provider_reported_cost,
        "agent_side_cost_accounting": {
            "schema": "papc.tau2.agent_side_cost_accounting.v1",
            "primary": (
                "cache_neutral_frozen_public_prices"
                if cache_neutral_cost is not None
                else "provider_reported_fallback"
            ),
            "cache_neutral_cost_usd": cache_neutral_cost,
            "provider_reported_cost_usd": provider_reported_cost,
            "pricing_snapshot_date": OPENAI_PUBLIC_PRICING_SNAPSHOT_DATE,
            "pricing_source": OPENAI_PUBLIC_PRICING_SOURCE,
            "roles": ["cheap", "strong"],
            "user_simulator_and_evaluator_excluded": True,
            "public_safe": True,
        },
        "projection_summary": _projection_summary(sidecar_rows),
    }


def tau2_runtime_ledger_from_results(
    *,
    results_payload: Mapping[str, Any],
    cheap_model: str,
    strong_model: str,
    sidecar_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Account transcript calls once and merge intercepted calls from sidecars."""

    ledger = RuntimeLedger()
    cheap_marker = str(cheap_model or "").lower()
    strong_marker = str(strong_model or "").lower()
    for simulation in _list_value(results_payload.get("simulations")):
        if not isinstance(simulation, Mapping):
            continue
        for message in _list_value(simulation.get("messages")):
            if not isinstance(message, Mapping):
                continue
            raw_data = _mapping_or_none(message.get("raw_data")) or {}
            cfrc_metadata = _mapping_or_none(raw_data.get("papc_cfrc"))
            if (
                cfrc_metadata is not None
                and cfrc_metadata.get("runtime_usage_accounted_in_transcript")
                is not True
            ):
                # Replica-generated graph messages are represented in sidecars.
                continue
            role = _runtime_role_for_tau_message(
                message,
                cheap_model=cheap_marker,
                strong_model=strong_marker,
            )
            if role is None:
                continue
            ledger.add_usage(
                role,
                usage_with_raw_cache_details(message),
                latency_seconds=_optional_float(message.get("generation_time_seconds")),
                cost_usd=_optional_float(message.get("cost")),
            )
        evaluation_time = _first_float(
            simulation.get("benchmark_evaluator_time_seconds"),
            simulation.get("evaluation_time_seconds"),
            simulation.get("evaluation_duration_seconds"),
            simulation.get("evaluator_duration_seconds"),
        )
        evaluation_cost = _optional_float(simulation.get("evaluation_cost"))
        reward_info = _mapping_or_none(simulation.get("reward_info"))
        if (
            reward_info is not None
            or evaluation_time is not None
            or evaluation_cost is not None
        ):
            ledger.add_usage(
                "evaluator",
                None,
                latency_seconds=evaluation_time,
                cost_usd=evaluation_cost,
            )
    for row in sidecar_rows:
        ledger.merge_public_summary(_mapping_or_none(row.get("runtime_ledger")))
    return ledger.public_summary()


def model_resource_summary(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "papc.tau2.model_resource_summary.v1",
        "roles": ["cheap", "strong", "user_simulator", "evaluator"],
        "per_arm": {
            name: {
                role: dict(
                    (_mapping_or_none(arm.get("runtime_ledger")) or {})
                    .get("per_role", {})
                    .get(role, {})
                )
                for role in ("cheap", "strong", "user_simulator", "evaluator")
            }
            for name, arm in arms.items()
        },
        "missing_fields_are_unavailable_not_zero": True,
        "raw_provider_payloads_excluded": True,
        "public_safe": True,
    }


def external_time_accounting(
    arms: Mapping[str, Mapping[str, Any]],
    *,
    parallel_elapsed_wall_seconds: float | None,
    max_concurrency: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    elapsed = _optional_float(parallel_elapsed_wall_seconds)
    concurrency = max(1, int(max_concurrency))
    for name, arm in arms.items():
        episode = _mapping_or_none(arm.get("episode_wall_clock_seconds")) or {}
        report[name] = {
            "schema": "papc.tau2.external_time_accounting.v1",
            "aggregate_episode_wall_seconds": episode.get("total_seconds"),
            "mean_episode_wall_seconds": episode.get("mean_seconds"),
            "episode_count": int(episode.get("count") or 0),
            "parallel_elapsed_wall_seconds": (
                round(elapsed, 6) if elapsed is not None else None
            ),
            "max_concurrency": concurrency,
            "parallel_capacity_seconds": (
                round(elapsed * concurrency, 6) if elapsed is not None else None
            ),
            "parallel_capacity_seconds_is_upper_bound_not_total": elapsed is not None,
            "missing_latency_fields_are_unavailable_not_zero": True,
            "public_safe": True,
        }
    return report


def _runtime_role_for_tau_message(
    message: Mapping[str, Any],
    *,
    cheap_model: str,
    strong_model: str,
) -> str | None:
    role = str(message.get("role") or "").lower()
    if role == "user":
        return "user_simulator"
    if role != "assistant":
        return None
    raw = _mapping_or_none(message.get("raw_data")) or {}
    cfrc_metadata = _mapping_or_none(raw.get("papc_cfrc")) or {}
    cfrc_runtime_role = str(cfrc_metadata.get("runtime_role") or "").lower()
    if cfrc_runtime_role in {"cheap", "strong"}:
        return cfrc_runtime_role
    model = str(raw.get("model") or message.get("model") or "").lower()
    if cheap_model and cheap_model in model:
        return "cheap"
    if strong_model and strong_model in model:
        return "strong"
    # Strong proposal calls are sidecar-only, so an unmarked ordinary
    # assistant transcript message belongs to the cheap prefix.
    return "cheap" if _mapping_or_none(message.get("usage")) is not None else None


def _projection_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reasons: Counter[str] = Counter()
    tools: Counter[str] = Counter()
    execute_count = 0
    strong_calls = 0
    graph_nodes = 0
    receipt_edges = 0
    terminal_edges = 0
    whole_graph_selected_count = 0
    receipt_errors = 0
    for row in rows:
        projection = _mapping_or_none(row.get("projection_result")) or {}
        if projection.get("execute") is True:
            execute_count += 1
        reasons[str(projection.get("reason") or "unknown")] += 1
        tools.update(
            str(name) for name in _list_value(projection.get("selected_tool_names"))
        )
        strong_calls += int(row.get("strong_call_count") or 0)
        proposal = _mapping_or_none(projection.get("proposal")) or {}
        graph_nodes += int(proposal.get("proposal_graph_node_count") or 0)
        receipt_edges += int(proposal.get("receipt_dependency_edge_count") or 0)
        terminal_edges += int(proposal.get("terminal_dependency_count") or 0)
        whole_graph_selected_count += int(proposal.get("whole_graph_selected") is True)
        receipts = _mapping_or_none(row.get("receipt_summary")) or {}
        receipt_errors += int(receipts.get("error") or 0)
    return {
        "schema": "papc.tau2.shadow_graph_projection_summary.v1",
        "decision_rows": len(rows),
        "execute_count": execute_count,
        "abstain_count": len(rows) - execute_count,
        "strong_call_count_sum": strong_calls,
        "reason_counts": dict(sorted(reasons.items())),
        "selected_tool_name_counts": dict(sorted(tools.items())),
        "proposal_graph_node_count_sum": graph_nodes,
        "receipt_dependency_edge_count_sum": receipt_edges,
        "terminal_dependency_count_sum": terminal_edges,
        "whole_graph_selected_count": whole_graph_selected_count,
        "proposal_receipt_error_count_sum": receipt_errors,
        "model_authored_relations": False,
        "public_safe": True,
    }


def _official_action_closure(
    simulations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    counts: dict[str, list[int]] = {
        "read": [0, 0],
        "write": [0, 0],
        "unknown": [0, 0],
        "db_match": [0, 0],
    }
    for simulation in simulations:
        reward_info = _mapping_or_none(simulation.get("reward_info")) or {}
        for check in _list_value(reward_info.get("action_checks")):
            if not isinstance(check, Mapping):
                continue
            bucket = str(check.get("tool_type") or "unknown").lower()
            if bucket not in {"read", "write"}:
                bucket = "unknown"
            counts[bucket][1] += 1
            counts[bucket][0] += int(check.get("action_match") is True)
        db_check = _mapping_or_none(reward_info.get("db_check"))
        if db_check is not None:
            counts["db_match"][1] += 1
            counts["db_match"][0] += int(db_check.get("db_match") is True)
    action_matched = sum(counts[key][0] for key in ("read", "write", "unknown"))
    action_total = sum(counts[key][1] for key in ("read", "write", "unknown"))
    return {
        "schema": "papc.tau2.official_action_closure.v1",
        **{
            key: {"matched": matched, "total": total}
            for key, (matched, total) in counts.items()
        },
        "all": {"matched": action_matched, "total": action_total},
    }


def _episode_wall_clock_summary(
    simulations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    durations = [
        value
        for row in simulations
        if (value := _optional_float(row.get("duration"))) is not None
    ]
    total = sum(durations)
    return {
        "schema": "papc.tau2.episode_wall_clock.v1",
        "count": len(durations),
        "total_seconds": round(total, 6) if durations else None,
        "mean_seconds": round(total / len(durations), 6) if durations else None,
    }


def _runtime_ledger_completeness(ledger: Mapping[str, Any]) -> dict[str, Any]:
    per_role = _mapping_or_none(ledger.get("per_role")) or {}
    required = ("cheap", "strong", "user_simulator", "evaluator")
    missing = [
        role
        for role in required
        if not (_mapping_or_none(per_role.get(role)) or {}).get("observed")
    ]
    return {
        "schema": "papc.tau2.runtime_ledger_completeness.v1",
        "required_roles": list(required),
        "missing_roles": missing,
        "all_required_roles_observed": not missing,
        "missing_fields_are_unavailable_not_zero": True,
    }


def _ledger_token_reconciliation(
    *,
    results_payload: Mapping[str, Any],
    sidecar_rows: Sequence[Mapping[str, Any]],
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    """Check that every token the run generated reached the ledger.

    A replica-authored message is kept out of the transcript tally and is
    expected to arrive through its sidecar row instead. When neither path
    records it the usage disappears silently and the reported cost is too low,
    so the two sources are added up here and compared with the ledger.
    """

    observed_input = observed_output = 0
    for simulation in _list_value(results_payload.get("simulations")):
        if not isinstance(simulation, Mapping):
            continue
        for message in _list_value(simulation.get("messages")):
            if not isinstance(message, Mapping):
                continue
            usage = _mapping_or_none(message.get("usage")) or {}
            totals = UsageAccumulator().add_usage(usage).as_dict()
            observed_input += int(totals["input_tokens"] or 0)
            observed_output += int(totals["output_tokens"] or 0)
    for row in sidecar_rows:
        per_role = (
            _mapping_or_none((_mapping_or_none(row.get("runtime_ledger")) or {}).get("per_role"))
            or {}
        )
        for role in ("cheap", "strong"):
            entry = _mapping_or_none(per_role.get(role)) or {}
            observed_input += int(entry.get("input_tokens") or 0)
            observed_output += int(entry.get("output_tokens") or 0)

    per_role = _mapping_or_none(ledger.get("per_role")) or {}
    ledger_input = sum(
        int((_mapping_or_none(entry) or {}).get("input_tokens") or 0)
        for entry in per_role.values()
    )
    ledger_output = sum(
        int((_mapping_or_none(entry) or {}).get("output_tokens") or 0)
        for entry in per_role.values()
    )
    input_delta = observed_input - ledger_input
    output_delta = observed_output - ledger_output
    return {
        "schema": "papc.tau2.runtime_ledger_token_reconciliation.v1",
        "transcript_and_sidecar_input_tokens": observed_input,
        "transcript_and_sidecar_output_tokens": observed_output,
        "ledger_input_tokens": ledger_input,
        "ledger_output_tokens": ledger_output,
        "unaccounted_input_tokens": input_delta,
        "unaccounted_output_tokens": output_delta,
        "reconciles": input_delta == 0 and output_delta == 0,
        "public_safe": True,
    }


def _provider_reported_agent_side_cost(
    ledger: Mapping[str, Any],
) -> float | None:
    per_role = _mapping_or_none(ledger.get("per_role")) or {}
    total = 0.0
    observed = False
    for role in ("cheap", "strong"):
        row = _mapping_or_none(per_role.get(role)) or {}
        if not row.get("observed"):
            continue
        if row.get("cost_available") is not True:
            return None
        total += float(row.get("cost_usd") or 0.0)
        observed = True
    return round(total, 6) if observed else None


def _cache_neutral_agent_side_cost(
    ledger: Mapping[str, Any],
    *,
    cheap_model: str,
    strong_model: str,
) -> float | None:
    per_role = _mapping_or_none(ledger.get("per_role")) or {}
    total = 0.0
    observed = False
    for role, model in (("cheap", cheap_model), ("strong", strong_model)):
        row = _mapping_or_none(per_role.get(role)) or {}
        if not row.get("observed"):
            continue
        if row.get("token_usage_available") is not True:
            return None
        input_tokens = row.get("input_tokens")
        output_tokens = row.get("output_tokens")
        if input_tokens is None or output_tokens is None:
            return None
        try:
            total += cache_neutral_cost_usd(
                model=model,
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                reasoning_tokens=row.get("reasoning_tokens"),
            )
        except ValueError:
            return None
        observed = True
    return round(total, 6) if observed else None


def _message_tool_calls(message: Any) -> list[Any]:
    value = _message_attr(message, "tool_calls")
    return value if isinstance(value, list) else []


def _tool_messages(message: Any) -> tuple[Any, ...]:
    rows = _message_attr(message, "tool_messages")
    if isinstance(rows, list):
        return tuple(rows)
    return (message,) if _message_attr_str(message, "role") == "tool" else ()


def _receipt_content_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value.strip()


def _tool_call_prompt_json(call: Any) -> dict[str, Any]:
    arguments = _message_attr(call, "arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    row = {
        "name": _message_attr_str(call, "name"),
        "arguments": dict(arguments) if isinstance(arguments, Mapping) else {},
    }
    call_id = _message_attr_str(call, "id")
    if call_id:
        row["tool_call_id"] = call_id
    return row


def _truncate_text(value: str, max_chars: int) -> str:
    if max_chars < 1:
        return ""
    if len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return value[:max_chars]
    return value[: max_chars - 3] + "..."


def _message_attr_str(message: Any, name: str) -> str:
    value = _message_attr(message, name)
    return str(value).strip() if value is not None else ""


def _message_attr(message: Any, name: str) -> Any:
    if isinstance(message, Mapping):
        return message.get(name)
    return getattr(message, name, None)


def _list_value(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _mapping_or_none(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _optional_float(value)
        if parsed is not None:
            return parsed
    return None


message_attr = _message_attr
message_attr_str = _message_attr_str
message_tool_calls = _message_tool_calls
mapping_or_none = _mapping_or_none
optional_float = _optional_float
tool_call_json = _tool_call_prompt_json


__all__ = [
    "CompiledGraphStep",
    "attach_metadata",
    "compact_visible_evidence",
    "compiled_graph_steps",
    "execute_on_replica",
    "external_time_accounting",
    "graph_step_receipt_matches",
    "json_safe_value",
    "llm_args",
    "llm_args_for_model",
    "litellm_model_name",
    "load_jsonl",
    "mapping_or_none",
    "message_attr",
    "message_attr_str",
    "message_tool_calls",
    "model_resource_summary",
    "optional_float",
    "retimestamp_delayed_message",
    "runtime_ledger_for_messages",
    "safe_agent_name",
    "arm_public_summary",
    "tau2_runtime_ledger_from_results",
    "tool_call_json",
    "write_jsonl_append",
]
